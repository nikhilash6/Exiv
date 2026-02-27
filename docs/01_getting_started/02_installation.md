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

* **Python 3.10+**
* **NVIDIA Drivers** (525+)
* **CUDA Toolkit 12.1+**

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
> **Note:** Not yet tested on this platform.


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
