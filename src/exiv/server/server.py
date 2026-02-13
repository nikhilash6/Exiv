import json
import os, sys, importlib
import asyncio
import threading
from glob import glob
import traceback
from typing import Any, Dict
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .app_core import App, TaskContext
from ..components.extension_registry import ExtensionRegistry
from .task_manager import RunRequest, ScriptResponse, ScriptStatus, TaskDetails, task_manager
from ..utils.logging import app_logger
from ..utils.file_path import FilePaths

APP_REGISTRY = {} # stores all the loaded apps

def load_apps_from_directory(directory: str = "apps"):
    """
    Scans the given directory for .py files, loads them, 
    and looks for an 'app' instance in them
    """
    # overriding via environment variable
    custom_dir = os.environ.get("EXIV_APPS_DIR")
    if custom_dir:
        apps_path = os.path.abspath(custom_dir)
    else:
        apps_path = os.path.join(os.getcwd(), directory)
    
    if not os.path.exists(apps_path):
        app_logger.warning(f"No 'apps' folder found at {apps_path}")
        return

    sys.path.append(apps_path)
    app_logger.debug(f"Scanning for apps in: {apps_path}")
    py_files = glob(os.path.join(apps_path, "*.py"))

    for file_path in py_files:
        module_name = os.path.basename(file_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "app") and isinstance(module.app, App):
                    APP_REGISTRY[module.app.name] = module.app
                    app_logger.debug(f"Loaded: {module.app.name}")
            except Exception as e:
                app_logger.error(f"Error loading {module_name}: {e}")

# Initialize Systems
# 1. Apps
load_apps_from_directory()
app_logger.info(f"Apps found: {[a for a, _ in APP_REGISTRY.items()]}")

# 2. Extensions
ExtensionRegistry.get_instance()

def process_task(task_id: str):
    def _update_task(status, progress, msg, output=None, data=None):
        assert 'status' in msg, f"Missing status in {msg}"
        msg = json.dumps(msg)
        task_manager.update_task(
            task_id,
            ScriptResponse(
                status=status,
                progress=progress,
                progress_message=msg,
                output=output,
                data=data,
            )
        )    
    
    def report_progress(progress: float, message: str = "Processing"):
        # to be used by the underlying script to report its progress
        _update_task(ScriptStatus.PROCESSING.value, progress, {"status": message})
    
    try:
        task_details = task_manager.get_task(task_id)
        app_name = task_details.app_name
        params = task_details.params
        app_def = APP_REGISTRY.get(app_name)
        
        app_logger.info(f"Processing task {task_id[-5:]}: {task_details.app_name}")
        _update_task(ScriptStatus.PROCESSING.value, 0, {"status": "Task Started"})

        if not app_def:
            app_logger.error(f"Error: App '{app_name}' not found in registry.")
            _update_task(
                ScriptStatus.FAILED.value, 
                0, 
                {"status": "Task Failed"}, 
                data={"err_message": f"App '{app_name}' not found"}
            )
            return
        
        params["context"] = TaskContext(report_progress)

        result = app_def.handler(**params)
        
        safe_result = result if isinstance(result, dict) else {}
        for o in app_def.outputs:
            if str(o.id) not in result.keys():
                app_logger.warning(f"Output ID:{o.id} is not in the result")
                
        _update_task(
            ScriptStatus.COMPLETED.value, 
            1, 
            {"status": "Task finished successfully"},
            output=safe_result,
            data=None
        )
    except Exception as e:
        app_logger.error(f"Exception occured: {e}")
        traceback.print_exc()
        _update_task(
            ScriptStatus.FAILED.value, 
            0,
            {"status": "Task Failed"}, 
            output=None,
            data={"err_message": str(e)}
        )

def start_worker(sync_mode=False):
    while True:
        task_id: str
        task_item = task_manager.task_queue.get()
        if task_item is None or task_item[0] is None:     # for stopping (funny thing, this is called 'poison pill')
            break
        
        task_id, _ = task_item
        process_task(task_id)
        task_manager.task_queue.task_done()
        
        if sync_mode: break                               # for cli, pkg import


app = FastAPI()

@app.get("/api/extensions")
def get_extensions():
    """
    Returns the metadata for ALL registered extensions.
    Includes ID, Name, Version, Capabilities, Inputs (Schema), Slot.
    """
    return ExtensionRegistry.get_instance().get_all_extensions_metadata()

@app.get("/api/apps")
def get_apps():
    return [app.model_dump(exclude={"handler"}) for app in APP_REGISTRY.values()]

@app.post("/api/apps/run")
async def run_app_endpoint(request: RunRequest):
    app_name = request.app_name
    user_params = request.params

    if app_name not in APP_REGISTRY:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=f"App '{app_name}' not found")
    
    target_app = APP_REGISTRY[app_name]
    
    clean_data = {}
    try:
        for key, input_def in target_app.inputs.items():
            val = user_params.get(key, input_def.default)
            clean_data[key] = input_def.validate_value(val)
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    task_id = task_manager.add_task(app_name=app_name, params=clean_data)
    
    return {"status": "queued", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_script_progress(task_id: str):
    status = task_manager.get_task_progress(task_id)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/api/outputs/{filename:path}")
async def get_output_file(filename: str):
    out_dir = FilePaths.OUTPUT_DIRECTORY
    file_path = os.path.join(out_dir, filename)
    
    # prevent directory traversal (e.g. "../filename")
    if not os.path.abspath(file_path).startswith(os.path.abspath(out_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)

@app.get("/api/outputs")
async def list_output_files(subfolder: str = None):
    out_dir = FilePaths.OUTPUT_DIRECTORY
    
    if subfolder:
        out_dir = os.path.join(out_dir, subfolder)
        # security check for subfolder
        if not os.path.abspath(out_dir).startswith(os.path.abspath(FilePaths.OUTPUT_DIRECTORY)):
             raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(out_dir):
        return []
        
    files = []
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            files.append({
                "filename": filename, 
                "extension": ext
            })
            
    return files

@app.websocket("/ws/status/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    update_freq = 0.5     # seconds
    await websocket.accept()
    if task_manager.get_task_progress(task_id) is None:
        await websocket.close(code=1008, reason="Task not found")
        return

    try:
        while True:
            progress = task_manager.get_task_progress(task_id)
            await websocket.send_json(progress)
            if progress['status'] in [ScriptStatus.COMPLETED.value, ScriptStatus.FAILED.value]:
                break
            await asyncio.sleep(update_freq)
            
    except WebSocketDisconnect:
        app_logger.info(f"WebSocket client disconnected for task {task_id}")
    finally:
        try:
            await websocket.close()
        except: pass

def run_server():
    worker_thread = threading.Thread(target=start_worker)
    worker_thread.daemon = True
    worker_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)