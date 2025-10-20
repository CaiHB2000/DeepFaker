# Auto-import all detector modules so their @register runs.
from importlib import import_module
import pkgutil, pathlib

_pkg_path = pathlib.Path(__file__).parent
for m in pkgutil.iter_modules([str(_pkg_path)]):
    if not m.ispkg:
        import_module(f"{__name__}.{m.name}")
