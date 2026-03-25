from setuptools import setup, find_packages
import pathlib
import re


def _read_version() -> str:
    init_py = pathlib.Path(__file__).parent / "turboquant" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise RuntimeError("Could not read __version__ from turboquant/__init__.py")
    return m.group(1)


__version__ = _read_version()

setup(
    name="turboquant-kv",
    version=__version__,
    description="Open-source implementation of Google TurboQuant for extreme KV-cache compression",
    author="hackimov + Grok (xAI)",
    url="https://github.com/hackimov/turboquant-kv",
    project_urls={
        "Source": "https://github.com/hackimov/turboquant-kv",
    },
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    extras_require={
        # Triton is optional. On Windows use `triton-windows` (PyPI); on Linux/macOS the `triton` package.
        "triton": [
            "triton-windows; platform_system == 'Windows'",
            "triton; platform_system != 'Windows'",
        ],
        "hf": ["transformers>=4.48.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)