# Contribution Guide

We welcome contributions to Exiv! Whether it's adding new models, writing better documentation, or fixing bugs, your help is appreciated.

## What Can You Contribute?
You can contribute in many ways, including but not limited to:
*   **Bug Fixes**: Identifying and resolving issues in the codebase.
*   **Feature Additions**: Implementing new capabilities or improving existing ones. e.g. dynamic VAE tiling, improving module offloading, etc.
*   **Performance Improvements**: Optimizing code to run faster or use fewer resources. e.g. writing custom triton code or fusing operations.
*   **Hardware Compatibility**: Expanding support for different hardware backends and devices.
*   **Model Additions**: Integrating new models into the platform.
*   **Documentation**: Writing better documentation, tutorials, or guides.
*   **Feedback**: Most importantly, you can provide feedback on what you like, what you don't like, and what you'd like to see in the future.

## Discussing Ideas
We highly recommend discussing your ideas before investing significant time in development:
*   **Join our Discord**: The best place to chat with the community and maintainers about your ideas. [Discord Link](https://discord.gg/4eFZuJDYXg)
*   **Create an Issue**: If you prefer, you can open an issue on GitHub to discuss your proposed changes.

## Getting More Involved
If you want to stick around and help maintain the project, dm me on Discord! If you aren't sure what to pick up, I can point you to some tasks in the backlog to help you get the hang of things.

## Development Workflow
1.  **Fork & Clone**: Fork the repository and clone it locally.
2.  **Environment Setup**: Install the package with development dependencies (`pip install -e .[dev]`).
3.  **Branching**: Create a feature branch (`git checkout -b feature/my-new-app`).
4.  **Testing**: Ensure you don't break existing functionality by running the test suite.
5.  **Pull Request**: Submit a PR with a clear description of the problem and the proposed fix.

## Code Style
*   Follow standard PEP-8 guidelines.
*   Keep `App` logic separate from `Core` engine logic. If you are modifying a core `Mixin` or `Hook`, ensure the change is broadly applicable, not just a hack for one specific App.