"""Compatibility shim for Lightning 1.8 with setuptools releases lacking pkg_resources."""

import sys

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    from pip._vendor import pkg_resources
    sys.modules["pkg_resources"] = pkg_resources
