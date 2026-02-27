# Tests

Testing is crucial to maintaining the stability of Exiv, especially when modifying low-level hooks or VRAM management strategies.

## Running Tests
Tests are located in the `tests/` directory and are written using `pytest`.

```bash
pytest tests/
```

## Coverage & Current State
*Note: The current test suite is somewhat bloated and does not provide 100% coverage. Active refactoring of the tests is in progress.* 

If you add a new core feature, please include at least a basic unit test verifying its functionality. For Apps, integration tests (running the app standalone with mock inputs) are preferred.