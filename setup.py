import os
from setuptools import setup


here = os.path.dirname(os.path.abspath(__file__))
about = {}
with open(os.path.join(here, "asre2e", "__version__.py")) as f:
    exec(f.read(), about)

install_requires = [
    "numpy",
    "torch",
    "torchaudio",
]


setup(
    name="asre2e",
    version=about["__version__"],
    url="https://github.com/panpan-wu/asre2e",
    author="Panpan Wu",
    author_email="wupanpan8@163.com",
    description="ASRE2E: end-to-end auto speech recognition",
    license="Apache Software License",
    packages=["asre2e"],
    install_requires=install_requires,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
    ],
)
