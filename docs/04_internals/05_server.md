# The API Server

While Exiv Apps can run standalone locally, the real power of Exiv lies in its centralized API server.

## Architecture
The server is located in `src/exiv/server/`. It is responsible for:
1.  **Handling Requests**: Listening for incoming JSON payloads defining generation tasks.
2.  **Task Manager** (`task_manager.py`): Queuing requests and delegating them to the active VRAM environment, ensuring multiple heavy tasks don't cause an Out Of Memory (OOM) crash simultaneously.
3.  **App Core** (`app_core.py`): Translating a server request into an execution of the relevant `App` handler.

## Starting the Server
```bash
exiv serve
```
