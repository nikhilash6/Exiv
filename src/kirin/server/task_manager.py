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
class ScriptRequest(BaseModel):
    filename: str
    git_url: str | None = None
    git_commit: str | None = None
    metadata: Dict | None = None                # for saved settings / inputs
    
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
    payload: ScriptRequest
    result: ScriptResponse
# ---------------------------------------------

class TaskManager:
    def __init__(self):
        # TODO: replace with a sqlite db maybe?
        self.task_dict: Dict[str, TaskDetails] = {}     # status and res is updated in place for now
        self.task_queue = Queue()

    def add_task(self, script_request: ScriptRequest):
        task_id = str(uuid.uuid4())
        self.task_dict[task_id] = TaskDetails(
            payload=script_request,
            result=ScriptResponse()
        )
        self.task_queue.put((task_id, script_request))
        return task_id
    
    def update_task(self, task_id, result: ScriptResponse):
        if not task_id in self.task_dict: raise RuntimeError(f"invalid task_id {task_id}")
        self.task_dict[task_id].result = result

    def get_task_progress(self, task_id: str):
        if task_id in self.task_dict:
            return self.task_dict[task_id].result.model_dump()
        else:
            return None
        
task_manager = TaskManager()