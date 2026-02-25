# Overview

Apps are high-level abstractions built on top of the Exiv engine. You can choose to integrate Exiv components in it or it can be a completely independent application. Apps also support custom UIs, so any UI that you create can be directly used in the App interface. You can read more about the UI system in the [Custom UIs](../03_apps/04_building_custom_app_ui.md).

## Core concepts
An App connects the core engine (models, hooks, conditionings, business logic) into an executable script. It consists of three main components:

1.  **Inputs**: Data like text prompts, base videos, or specific conditioning formats.
2.  **Outputs**: Images, videos, or raw tensor structures.
3.  **Handler**: The Python logic that maps inputs to outputs using the model pipeline.

Apps can be run standalone as CLI scripts or served through the centralized task manager.