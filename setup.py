from setuptools import setup, find_packages

setup(
    name="ds4finance",
    version="0.1.3",
    description="A collection of data science tools for finance",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Luis Silva",
    author_email="luis_paulo_silva@hotmail.com",
    url="https://github.com/LuisSousaSilva/ds4finance",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
)


