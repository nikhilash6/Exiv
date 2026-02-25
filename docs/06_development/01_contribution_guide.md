# Contribution Guide

We welcome contributions to Exiv! Whether it's adding new models, writing better documentation, or fixing bugs, your help is appreciated.

## Development Workflow
1.  **Fork & Clone**: Fork the repository and clone it locally.
2.  **Environment Setup**: Install the package with development dependencies (`pip install -e .[dev]`).
3.  **Branching**: Create a feature branch (`git checkout -b feature/my-new-app`).
4.  **Testing**: Ensure you don't break existing functionality by running the test suite.
5.  **Pull Request**: Submit a PR with a clear description of the problem and the proposed fix.

## Code Style
*   Follow standard PEP-8 guidelines.
*   Keep `App` logic separate from `Core` engine logic. If you are modifying a core `Mixin` or `Hook`, ensure the change is broadly applicable, not just a hack for one specific App.