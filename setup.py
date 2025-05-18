from setuptools import setup, find_packages

setup(
    name="network",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A neural network simulator with various node types and kernels",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/network",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 