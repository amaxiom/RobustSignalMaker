"""Compatibility shim for legacy pip/setuptools (editable installs).

pyproject.toml is the authoritative package definition. Toolchains with
setuptools < 61 cannot read its [project] table (and pip < 21.3 cannot do a
pyproject-only editable install), so on those toolchains this shim duplicates
the minimum metadata, reading the version from robustsignalmaker/__init__.py
so the single source of truth is preserved. Modern toolchains read
pyproject.toml and this shim passes nothing.
"""
import re
from pathlib import Path

import setuptools

kwargs = {}
if int(setuptools.__version__.split(".")[0]) < 61:
    version = re.search(
        r'^__version__ = "([^"]+)"',
        (Path(__file__).parent / "robustsignalmaker" / "__init__.py").read_text(),
        re.MULTILINE,
    ).group(1)
    kwargs = dict(
        name="robustsignalmaker",
        version=version,
        description="NaN-aware, leakage-free stability selection for scientific signals and spectra",
        author="Amanda S Barnard",
        packages=["robustsignalmaker"],
        package_data={"robustsignalmaker": ["py.typed"]},
        python_requires=">=3.9",
        install_requires=["numpy>=1.24", "scipy>=1.10", "scikit-learn>=1.3"],
    )

setuptools.setup(**kwargs)
