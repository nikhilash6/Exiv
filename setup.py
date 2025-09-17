from typing import List
from setuptools import setup, find_packages

def read_requirements_file(file_name: str) -> List[str]:
    with open(f"./{file_name}.txt", "r", encoding="utf-8") as file:
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
    name="kirin",
    version="0.1",
    description="Fastest and lightest Gen AI backend!",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Piyush Kumar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # package_data={"test_pkg": ["data.txt"]},
    install_requires=main_deps,
    extras_require={
        'dev': dev_deps,
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "kirin = kirin.main:cli"
        ]
    }
)