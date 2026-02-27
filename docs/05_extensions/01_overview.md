# Overview

Extensions are one of the most powerful ways to customize or enhance Exiv. At the moment it allows you to completely modify any method or inject new functionality in the core engine. 

## What is an Extension?
While an **App** uses the core engine to fulfill a high-level task (like text-to-video), an **Extension** adds new low-level features to the engine itself. You can check some examples below to get a better idea of the difference between an App and an Extension.

### `github.com/piyushK52/exiv_dwpose`
A complete example of integrating an external pose-estimation model (DWPose) into Exiv as a specialized pre-processor for image or video conditioning.

### `github.com/piyushK52/exiv_matanyone`
Demonstrates how to wrap complex third-party tools (MatAnyone) for advanced masking and matting tasks within an Exiv generation pipeline.

## Registering Extensions
The `src/exiv/components/extension_registry.py` handles the discovery and loading of extensions placed in the `extensions/` directory. By adhering to the extension format, your plugin becomes instantly accessible to any App running on the server.

