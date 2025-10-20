# Auto-import all probe modules in this package so their @register runs.
from importlib import import_module
import pkgutil, pathlib

_pkg_path = pathlib.Path(__file__).parent
for m in pkgutil.iter_modules([str(_pkg_path)]):
    # import every .py module under this package (skip subpackages if any)
    if not m.ispkg:
        import_module(f"{__name__}.{m.name}")
