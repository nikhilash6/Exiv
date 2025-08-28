import os
import iglob
import re

def create_sanitized_path(file_path):
    filename = os.path.join(file_path, "img_{idx}.jpg")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        idx = 0
    else:
        # if same filename is already present then increase the idx
        fns = [fn for fn in iglob(filename.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
            
    return idx