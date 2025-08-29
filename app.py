import importlib.util
import importlib.metadata
from typing import Tuple, Union
from collections import defaultdict
import inspect



def _is_package_available(pkg_name: str, get_dist_name: bool = True) -> Tuple[bool, str]:
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        try:
            package_map = importlib.metadata.packages_distributions()
        except Exception as e:
            package_map = defaultdict(list)
            if isinstance(e, AttributeError):
                try:
                    # Fallback for Python < 3.10
                    for dist in importlib.metadata.distributions():
                        _top_level_declared = (dist.read_text("top_level.txt") or "").split()
                        _infered_opt_names = {
                            f.parts[0] if len(f.parts) > 1 else inspect.getmodulename(f) for f in (dist.files or [])
                        } - {None}
                        _top_level_inferred = filter(lambda name: "." not in name, _infered_opt_names)
                        for pkg in _top_level_declared or _top_level_inferred:
                            package_map[pkg].append(dist.metadata["Name"])
                except Exception as _:
                    pass

        try:
            if get_dist_name and pkg_name in package_map and package_map[pkg_name]:
                if len(package_map[pkg_name]) > 1:
                    print(
                        f"Multiple distributions found for package {pkg_name}. Picked distribution: {package_map[pkg_name][0]}"
                    )
                pkg_name = package_map[pkg_name][0]
            pkg_version = importlib.metadata.version(pkg_name)
            print(f"Successfully imported {pkg_name} version {pkg_version}")
        except (ImportError, importlib.metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version

ch = ["torch", "numpy", "protobuf"]
for c in ch:
    print(_is_package_available(c))