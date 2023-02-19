from setuptools import setup, find_packages# List of requirements
requirements = []  # This could be retrieved from requirements.txt# Package (minimal) configuration
setup(
    name = "mental",
    author = "Aite Kuo",
    author_email = "robinsa87@gmail.com",
    version = "1.0.0",
    keywords = "Deep learning depression early detection pytorch",
    description = "",
    packages = [ "mental", "preprocess"],
    package_dir = {
        "": "src",
        "preprocess": "src",
    },
    install_requires = requirements
)