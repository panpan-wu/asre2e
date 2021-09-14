import os
from setuptools import setup


install_requires = [
    "torch",
    "torchaudio",
]
tests_require = [
    "pytest",
]

setup(
    name="asre2e",
    version="1.0.0",
    url="https://github.com/panpan-wu/asre2e",
    author="Panpan Wu",
    author_email="wupanpan8@163.com",
    description="ASRE2E: end-to-end auto speech recognition",
    license="Apache Software License",
    packages=["asre2e"],
    install_requires=install_requires,
    tests_require=tests_require,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
    ],
)
