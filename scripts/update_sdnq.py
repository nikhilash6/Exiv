import requests
import os

FILES = [
    "common.py", "dequantizer.py", "forward.py", 
    "packed_int.py", "sdnext.py", "triton_mm.py"
]

LAYERS_FILES = [
    "layers/linear/linear_int8.py", "layers/linear/forward.py", 
    "layers/linear/linear_fp8.py", "layers/linear/linear_fp16.py",
    "layers/conv/conv_int8.py", "layers/conv/forward.py",
    "layers/conv/conv_fp8.py", "layers/conv/conv_fp16.py",
]

BASE_URL = "https://raw.githubusercontent.com/Disty0/sdnq/main/src/sdnq/"
TARGET_DIR = "src/exiv/quantizers/sdnq_lib/"

def download_file(rel_path):
    url = BASE_URL + rel_path
    target_path = os.path.join(TARGET_DIR, rel_path)
    print(f"Updating {rel_path}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"Failed to fetch {rel_path}: {e}")

if __name__ == "__main__":
    for f in FILES + LAYERS_FILES:
        download_file(f)
    print("SDNQ Lib Updated.")