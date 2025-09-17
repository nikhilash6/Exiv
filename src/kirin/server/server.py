import threading
import time
from fastapi import FastAPI, HTTPException

from .task_manager import ScriptRequest, task_manager
from ..utils.logging import app_logger


def start_worker():
    while True:
        task_id, task_details = task_manager.task_queue.get()
        if task_id is None:     # for stopping
            break
        app_logger.info(f"Processing task {task_id}: {task_details}")
        # simulate a long-running task
        time.sleep(5)
        task_manager.results[task_id] = {
            "status": "completed",
            "output": f"Result for {task_details.get('prompt', 'N/A')}"
        }
        task_manager.task_queue.task_done()


app = FastAPI()

@app.post("/queue")
async def queue_script(task_request: ScriptRequest):
    task_id = task_manager.add_task(task_request)
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