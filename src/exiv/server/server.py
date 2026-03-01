import json
import os, sys, importlib
import asyncio
import threading
from glob import glob
import traceback
from typing import Any, Dict
import shutil
import uuid
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
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
    
    # scan for:
    # 1. apps/*.py
    # 2. apps/*/app.py
    # 3. apps/*/__init__.py
    
    py_files = glob(os.path.join(apps_path, "*.py"))
    py_files.extend(glob(os.path.join(apps_path, "*", "app.py")))
    py_files.extend(glob(os.path.join(apps_path, "*", "__init__.py")))

    for file_path in py_files:
        filename = os.path.basename(file_path)
        app_dir = os.path.dirname(file_path)
        
        if filename in ["app.py", "__init__.py"]:
            # apps/my_app/app.py -> module name: my_app
            module_name = os.path.basename(app_dir)
        else:
            # apps/my_app.py -> module name: my_app
            module_name = filename.replace(".py", "")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "app") and (isinstance(module.app, App) or type(module.app).__name__ == "App"):
                    # frontend assets (dist/index.js, dist/style.css)
                    dist_path = os.path.join(app_dir, "dist")
                    if os.path.exists(dist_path):
                        js_path = os.path.join(dist_path, "index.js")
                        if os.path.exists(js_path):
                            # absolute path to dist folder
                            module.app.asset_root = dist_path
                            assets = {"js": "index.js"}
                            css_path = os.path.join(dist_path, "style.css")
                            if os.path.exists(css_path):
                                assets["css"] = "style.css"
                            
                            module.app.frontend_assets = assets
                            app_logger.debug(f"  -> Found frontend for {module.app.name}")

                    if module.app.name in APP_REGISTRY:
                        app_logger.warning(f"Warning: Multiple apps found with the same name '{module.app.name}'. This can lead to unexpected behaviour.")

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
    
    def report_progress(progress: float, message = "Processing"):
        # to be used by the underlying script to report its progress
        msg_dict = message if isinstance(message, dict) else {"status": message}
        _update_task(ScriptStatus.PROCESSING.value, progress, msg_dict)
    
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

        result = app_def.invoke(**params)
        
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

import hashlib

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file to the server's temp directory and returns the absolute path.
    Uses SHA256 hash of content to avoid redundant copies.
    """
    try:
        # read content to hash it
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        
        # temp directory for upload
        upload_dir = os.path.join(FilePaths.OUTPUT_DIRECTORY, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # use hash + original extension to keep it unique but identifiable
        _, ext = os.path.splitext(file.filename)
        filename = f"{file_hash}{ext}"
        file_path = os.path.abspath(os.path.join(upload_dir, filename))
        
        if os.path.exists(file_path):
            app_logger.info(f"File already exists (hash match): {file_path}")
            return {"status": "success", "file_path": file_path}

        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        app_logger.info(f"File uploaded successfully to: {file_path}")
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        app_logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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

@app.get("/api/apps/{app_name}/assets/{filename:path}")
async def get_app_asset(app_name: str, filename: str):
    if app_name not in APP_REGISTRY:
        raise HTTPException(status_code=404, detail="App not found")
    
    app = APP_REGISTRY[app_name]
    if not app.frontend_assets:
        raise HTTPException(status_code=404, detail="App has no frontend assets")

    # security check: filename must be allowed
    allowed_files = app.frontend_assets.values()
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="Access denied to this file")

    if not app.asset_root:
         raise HTTPException(status_code=500, detail="App asset root not configured")
         
    file_path = os.path.join(app.asset_root, filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Asset not found")
        
    return FileResponse(file_path)

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
        pass
    finally:
        try:
            await websocket.close()
        except: pass

def run_server():
    worker_thread = threading.Thread(target=start_worker)
    worker_thread.daemon = True
    worker_thread.start()

    import uvicorn
    # Suppress uvicorn's default access and connection logging 
    # so polling doesn't spam the console.
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, log_level="warning")