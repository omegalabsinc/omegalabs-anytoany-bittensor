from setuptools import setup, find_packages

setup(
    name="validator",
    version="0.1.0",
    packages=find_packages(include=['neurons*', 'evaluation*', 'models*']),
    python_requires="==3.10.*"  # Matching pyproject.toml python version requirement
)
