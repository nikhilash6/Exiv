<div align="center">
  <h1>
    <img src="https://raw.githubusercontent.com/piyushK52/Exiv/main/public/favicon.svg" alt="" width="45" style="vertical-align: middle; margin-right: 10px;" /> Exiv
  </h1>
</div>

Exiv is a fast, lightweight, and highly extensible AI backend engine designed for running, patching, and serving generative models and orchestrating workflows efficiently. Please check the [docs](https://exiv.pages.dev/) for detailed guides, API references, and examples.

If you want a GUI to interact with Exiv, check it out [here](https://github.com/piyushK52/Exiv-UI).

<div align="center">
  <!-- TODO: Add Explainer Video Link Here -->
  <a href="#">
    <img src="https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg" alt="Exiv Explainer Video" width="600"/>
  </a>
</div>

## ✨ Feature Summary

🐍 **Pure Python:** Have complete control to modify any part of the framework. No complex compilation steps or opaque C++ bindings getting in your way.

🔪 **Cutting Edge Models:** Built to support state-of-the-art open-source GenAI models with optimized inference.

♻️ **Low VRAM support:** Exiv intelligently manages your VRAM and system memory, offloading models dynamically so you only use what you need, maximizing performance even on constrained hardware.

🔌 **Plug & Play Extensions:** Easily extend the engine's core capabilities by writing custom plugins. All features can be exposed via our JSON API server or embedded directly into your Python scripts.

🚀 **Modular & Extensible:** Core architecture is broken into conditionings and hooks, making it incredibly simple to build, debug, and extend.

☁️ **Server & Workflow Ready:** Ships with a built-in API server to easily host models and orchestrate complex generative workflows/logic.

🎨 **Your Apps, Your Way:** Seamlessly combine custom backend logic with your custom UIs to build exactly the generative tools you want.

## 🛠️ Installation

For a full guide and platform-specific requirements, please refer to our [Installation Documentation](https://exiv.pages.dev/docs/getting-started/installation).

### Option A: Clone from Source (Recommended for hacking and dev)
If you want to modify the source code, build extensions, or contribute to Exiv:
```bash
git clone git@github.com:piyushK52/Exiv.git
cd Exiv

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including dev tools)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

### Option B: Install via PIP (Recommended for feature integration)
If you just want to use the stable version as a Python package and don't need to tinker with the codebase:
```bash
pip install exiv
```

## ❓ FAQs

**Q: Why Exiv?**  
**A:** It's a modular and extensible engine written in pure Python, best for devs to build on top of or integrate into their own apps. The App building feature provides complete control to anyone to build their own apps on top of Exiv. Vibe coded UIs are supported as well. The design philosophy makes it extremely easy to understand the codebase and add new functionalities - everything is either a conditioning or a hook!

**Q: How is this different from Diffusers?**  
**A:** Exiv is extremely lean. Because it is built from scratch it does away with a LOT of dependencies that Diffusers has. Exiv is also deeply modular and highly extensible by design. Instead of monolithic pipelines, Exiv's architecture leverages granular hooks and conditionings, giving developers surgical control over model execution without having to rewrite or monkey-patch core components.

**Q: How is this different from ComfyUI?**  
**A:** Exiv is written in pure Python, meaning you construct, debug, and execute your logic using standard Python code rather than a node-based visual interface. This gives developers standard programmatic tools and testing capabilities that visual interfaces often abstract away. (Going forward there will be significant feature additions that will make the differentiation even more apparent).

**Q: What's the roadmap?**  
**A:** Primary focus is to continuously integrate the latest models, but more importantly, we are dedicated to building opinionated research methods and cutting-edge inference optimizations natively into the engine.

**Q: Why contribute to Exiv?**  
**A:** Since Exiv is still in its early stages, it's incredibly easy to grasp the core concepts of the codebase and make meaningful contributions right away. Because the project has a strong research focus, implementing novel methods or experimental optimizations here is a fantastic way to learn. Furthermore, your PRs are highly likely to be merged—I am very responsive, actively maintaining the project, and always excited to welcome new contributors!