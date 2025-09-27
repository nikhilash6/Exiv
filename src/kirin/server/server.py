import os, sys, importlib
import asyncio
import threading
import time
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .task_manager import ScriptRequest, ScriptResponse, ScriptStatus, TaskDetails, task_manager
from ..utils.logging import app_logger


def process_task(task_id: str, script_request: ScriptRequest):
    app_logger.info(f"Processing task {task_id[-5:]}: {script_request.filename}")
    try:
        # get script path
        script_path = os.path.abspath(script_request.filename)
        script_dir = os.path.dirname(script_path)
        script_name = os.path.splitext(os.path.basename(script_path))[0]

        # add the script's directory to the python path to handle relative imports
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # import the script as a module
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)

        # executing the 'main' function
        if hasattr(script_module, 'main'):
            script_module.main(script_request.metadata)
        else:
            raise RuntimeError(f"Script {script_request.filename} does not have a main function.")

        progress_percent = 1
        task_manager.update_task(
            task_id,
            ScriptResponse(
                status=ScriptStatus.PROCESSING.value,
                progress=progress_percent,
                progress_message=f"Completed step {1} of {1}"
            )
        )

        task_manager.update_task(
            task_id, 
            ScriptResponse(
                status=ScriptStatus.COMPLETED.value,
                progress=1.0,
                progress_message="Task finished successfully",
                output={"output1": "output_file.png"},
                data=None
            )
        )
    except Exception as e:
        app_logger.error(f"Exception occured: {e}")
        task_manager.update_task(
            task_id, 
            ScriptResponse(
                status=ScriptStatus.FAILED.value,
                output=None,
                data={"err_message": str(e)}
            )
        )

def start_worker(sync_mode=False):
    while True:
        task_id: str
        task_details: ScriptRequest
        task_id, task_details = task_manager.task_queue.get()
        if task_id is None:     # for stopping
            break
        process_task(task_id, task_details)
        task_manager.task_queue.task_done()
        
        if sync_mode: break     # for cli, pkg import


app = FastAPI()

@app.post("/queue")
async def queue_script(script_request: ScriptRequest):
    task_id = task_manager.add_task(script_request)
    return {"message": "Task received", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_script_progress(task_id: str):
    status = task_manager.get_task_progress(task_id)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.websocket("/ws/status/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    update_freq = 1     # seconds
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
        await websocket.close()

def run_server():
    worker_thread = threading.Thread(target=start_worker)
    worker_thread.daemon = True
    worker_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)