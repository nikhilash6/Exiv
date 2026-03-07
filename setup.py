import os
from typing import List
from setuptools import setup, find_packages

CWD = os.path.abspath(os.path.dirname(__file__))

def read_requirements_file(file_name: str) -> List[str]:
    with open(os.path.join(CWD, f"{file_name}.txt"), "r", encoding="utf-8") as file:
        requirements = file.readlines()

    res: List[str] = []
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        res.append(line)
    return res

main_deps = read_requirements_file("requirements")
dev_deps = read_requirements_file("requirements-dev")

setup(
    name="exiv",
    version="0.2",
    description="Modular & Extensible Gen AI backend!",
    long_description=open(os.path.join(CWD, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Piyush Kumar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=main_deps,
    extras_require={
        'dev': dev_deps,
    },
    package_data={
        "exiv": ["data/registry/*.json"],
    },
    entry_points={
        "console_scripts": [
            "exiv = exiv.main:cli"
        ]
    }
)