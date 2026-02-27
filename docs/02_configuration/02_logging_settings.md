# Logging Settings

Exiv provides a robust logging mechanism via `src/exiv/utils/logging.py` to trace system behavior.

## Log Levels
You can set the verbosity of logs using the `log_level` environment variable. 

*   **`log_level=1`** : Only `ERROR` + `WARNING` (Silent mode)
*   **`log_level=2`** : Only `WARNING`
*   **`log_level=3`** : (Default) `INFO` + `WARNING` + `ERROR` + `CRITICAL`
*   **`log_level=4`** : `DEBUG` (debug logs can get quite large)

**Example:**
```bash
export log_level=4
python apps/test.py
```