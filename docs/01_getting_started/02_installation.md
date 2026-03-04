# Installation

Follow these steps to set up the Exiv backend on your machine.

Select your operating system and device to see tailored instructions:

<!-- install-grid -->
<!-- option: Operating System = Linux | Windows | macOS -->
<!-- option: Device = NVIDIA | AMD | Apple Silicon -->
<!-- constraint: Operating System = Linux -> Device = NVIDIA | AMD -->
<!-- constraint: Operating System = Windows -> Device = NVIDIA | AMD -->
<!-- constraint: Operating System = macOS -> Device = Apple Silicon -->

<!-- section: linux-nvidia -->
## 1. Prerequisites

* **Python 3.10+** — check with `python3 --version`. If not installed, use your distro's package manager (e.g., `sudo apt install python3 python3-venv python3-pip` on Ubuntu/Debian) or download from [python.org](https://www.python.org/downloads/).
* **Git** — check with `git --version`. Install via `sudo apt install git` or from [git-scm.com](https://git-scm.com/downloads/linux).
* **NVIDIA Drivers** (525+) — install via `sudo apt install nvidia-driver-535` or from [NVIDIA's driver page](https://www.nvidia.com/en-us/drivers/).
* **CUDA Toolkit 12.1+** — install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). Verify with `nvcc --version`.
* **FFmpeg** *(optional, for saving metadata to output files)* — install via `sudo apt install ffmpeg`.

## 2. Clone the Repository

```bash
git clone git@github.com:piyushK52/Exiv.git
cd Exiv
```

## 3. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121    # if not already installed
pip install -e .[dev]       # [dev] installs development dependencies like pytest
```

## (Alternate) PIP Package
Use this method if you just want the stable version as a python package and don't want to tinker with the codebase.
```bash
pip install exiv
```
<!-- /section -->

<!-- section: linux-amd -->
> **Note:** Not yet tested on this platform.

<!-- /section -->

<!-- section: linux-cpu -->
> **Note:** Not yet tested on this platform.


<!-- /section -->

<!-- section: windows-nvidia -->
## 1. Prerequisites

### Python 3.10+
1. Download the installer from [python.org](https://www.python.org/downloads/) (Python 3.10 or newer).
2. Run the `.exe` installer. **Important:** check the **"Add python.exe to PATH"** box on the first screen before clicking Install.
3. Verify by opening **Command Prompt** or **PowerShell** and running:
   ```
   python --version
   pip --version
   ```

### Git
1. Download [Git for Windows](https://git-scm.com/downloads/win) and run the installer.
2. During installation, select **"Use Git from the Windows Command line and also from 3rd-party software"** to add Git to your PATH.
3. Verify: `git --version`

### CUDA Toolkit 12.1+
1. Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) (select Windows → your architecture).
2. Make sure you have up-to-date NVIDIA GPU drivers.
3. Verify: `nvcc --version`

### FFmpeg *(optional)*
Required only for saving metadata to output files. Download from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows), extract, and add the `bin` folder to your system PATH.

> **Note:** Do not install Exiv in folders that require admin permissions, hidden folders (starting with `.`), or synced folders like OneDrive — these can cause unexpected issues.

## 2. Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```
git clone git@github.com:piyushK52/Exiv.git
cd Exiv
```

## 3. Installation

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

> **Tip:** If using PowerShell and you get an "execution policy" error when activating the venv, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first.

## (Alternate) PIP Package
Use this method if you just want the stable version as a python package and don't want to tinker with the codebase.
```
pip install exiv
```
<!-- /section -->

<!-- section: windows-amd -->
> **Note:** Not yet tested on this platform.


<!-- /section -->

<!-- section: windows-cpu -->
> **Note:** Not yet tested on this platform.


<!-- /section -->

<!-- section: macos-apple silicon -->
> **Note:** Not yet tested on this platform.


<!-- /section -->

<!-- section: macos-cpu -->
> **Note:** Not yet tested on this platform.


<!-- /section -->

<!-- section: default -->
> **Note:** Not yet tested on this platform.


<!-- /section -->
