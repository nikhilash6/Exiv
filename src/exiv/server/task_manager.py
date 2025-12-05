from dataclasses import dataclass
import uuid
from typing import Dict, Any
from pydantic import BaseModel
from queue import Queue

from ..utils.enum import ExtendedEnum


class ScriptStatus(ExtendedEnum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    FAILED = 'failed'
    COMPLETED = 'completed'

# -------------- Serializers ----------------- 


class ServerResponse(BaseModel):
    status: int
    message: str
    data: Any | None = None
    
class ScriptResponse(BaseModel):
    status: str = ScriptStatus.PENDING.value
    output: Dict | None = None                  # output file locations
    data: Any | None = None                     # time taken, memory usage, err msg etc..
    progress: float = 0.0
    progress_message: str = ""                      # current component being processed

@dataclass
class TaskDetails:
    app_name: str
    params: Any
    result: ScriptResponse
# ---------------------------------------------

class TaskManager:
    def __init__(self):
        # TODO: replace with a sqlite db maybe?
        self.task_dict: Dict[str, TaskDetails] = {}     # task_dict is like an in-mem db
        self.task_queue: Queue[str] = Queue()

    def add_task(self, app_name: str, params: Any) -> str:
        task_id = str(uuid.uuid4())
        self.task_dict[task_id] = TaskDetails(
            app_name=app_name,
            params=params,
            result=ScriptResponse()
        )
        self.task_queue.put((task_id, app_name))
        return task_id
    
    def get_task(self, task_id):
        if task_id == None: return None
        return self.task_dict.get(task_id, None)
    
    def update_task(self, task_id, result: ScriptResponse):
        if not task_id in self.task_dict: raise RuntimeError(f"invalid task_id {task_id}")
        self.task_dict[task_id].result = result

    def get_task_progress(self, task_id: str) -> Dict:
        if task_id in self.task_dict:
            d = self.task_dict[task_id].result.model_dump()
            print("task output: ", d)
            return d
        else:
            return None
        
task_manager = TaskManager()