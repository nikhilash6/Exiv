from setuptools import setup, find_packages

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
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "show_dummy_data = kirin.main:print_stuff"
        ]
    }
)