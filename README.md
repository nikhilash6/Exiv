<div align="center">
  <h1>
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/e1c60da9-d953-43b6-b353-ce0c8f12c68c">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/64a8424d-8a7a-4873-8840-35666bdff415">
      <img src="https://github.com/user-attachments/assets/64a8424d-8a7a-4873-8840-35666bdff415" alt="" width="20" style="vertical-align: text-bottom; margin-right: 10px;" />
    </picture>
    <span style="vertical-align: middle;">Exiv</span>
  </h1>
</div>

Exiv is a fast, lightweight, and highly extensible AI backend engine designed for running, patching, and serving generative models and orchestrating workflows efficiently. Please check the [docs](https://exiv.pages.dev/) for detailed guides, API references, and examples.

> [!IMPORTANT]
> **Exiv is currently in early beta.** It's evolving fast—if you want to help shape the direction or contribute, check out the [Contribution Guide](CONTRIBUTING.md).

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

🪶 **Low VRAM support:** Exiv intelligently manages your VRAM and system memory, offloading models dynamically so you only use what you need, maximizing performance even on constrained hardware.

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
If you just want to use the stable version as a Python package and don't need to tinker with the codebase. In-built apps are not available in this mode.
```bash
pip install exiv
```

## 🚀 Quick Start

The fastest way to verify your installation and see Exiv in action is to run the simple text-to-video application. Note that this will require 20GB+ of model weights to be downloaded and is not available in the package mode.

```bash
# Run the simple T2V app with a custom prompt
python apps/simple_t2v.py --prompt "Cinematic shot of a Golden Retriever sprinting through a sunny park, 8k, motion blur."
```

Once the generation is complete, the path to the output video will be printed in your terminal.

## ❓ FAQs

**Q: Why Exiv?**  
**A:** It's a modular and extensible engine written in pure Python, best for devs to build on top of or integrate into their own apps. The App building feature provides complete control to anyone to build their own apps on top of Exiv. Vibe coded UIs are supported as well. The design philosophy makes it extremely easy to understand the codebase and add new functionalities - everything is either a conditioning or a hook!

**Q: How is this different from Diffusers?**  
**A:** While Diffusers is an incredibly robust and battle-tested library, it can feel opaque. Exiv is built for pure extensibility and a clean "whitebox" developer experience. Diffusers spreads its core logic across multiple massive libraries, whereas Exiv keeps everything—from memory offloading to zero-init logic—centralized in one lean codebase so you never have to go hunting to understand how things work. Instead of rigid monolithic pipelines and messy ad-hoc patching, Exiv uses granular hooks and a dedicated extension registry, giving you plug-and-play, surgical control over every step of the generation process.

**Q: How is this different from ComfyUI?**  
**A:** Exiv is written in pure Python, meaning you construct, debug, and execute your logic using standard Python code rather than a node-based visual interface. This gives developers standard programmatic tools and testing capabilities that visual interfaces often abstract away. (Going forward there will be significant feature additions that will make the differentiation even more apparent).

**Q: What's the roadmap?**  
**A:** Primary focus is to continuously integrate the latest models, but more importantly, we are dedicated to building opinionated research methods and cutting-edge inference optimizations natively into the engine.

**Q: Why contribute to Exiv?**  
**A:** Since Exiv is still in its early stages, it's incredibly easy to grasp the core concepts of the codebase and make meaningful contributions right away. Because the project has a strong research focus, implementing novel methods or experimental optimizations here is a fantastic way to learn. Furthermore, your PRs are highly likely to be merged—I am very responsive, actively maintaining the project, and always excited to welcome new contributors!

## 🙌 Acknowledgements

A huge thank you to the open-source projects and developers whose incredible work laid the foundation for the AI image/video generation space. Exiv draws inspiration (and occasionally some math/logic adaptations) from:
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [Automatic1111 (A1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Forge WebUI](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

And a special thanks to the broader **OSS AI community** for continuously pushing the boundaries of what is possible.

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. 

### What this means:
- **Open Source Derivatives:** Any modifications or derivative works based on this project must also be distributed under the GPLv3 license and made open source.
- **Attribution and Credit:** You must preserve all original copyright notices. Derivative works and distributions must prominently reference and give credit to **Exiv**. 

For the full license text, please see the [LICENSE](./LICENSE) file in the root of this repository.
