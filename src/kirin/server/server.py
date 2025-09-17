import threading
import time
from fastapi import FastAPI, HTTPException

from .task_manager import ScriptRequest, ScriptResponse, ScriptStatus, task_manager
from ..utils.logging import app_logger


def process_task(task_id, task_details):
    app_logger.info(f"Processing task {task_id}: {task_details}")
    try:
        # simulate a long-running task
        time.sleep(2)
        task_manager.update_task(
            task_id, 
            ScriptResponse(
                status=ScriptStatus.COMPLETED.value,
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

def run_server():
    worker_thread = threading.Thread(target=start_worker)
    worker_thread.daemon = True
    worker_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)