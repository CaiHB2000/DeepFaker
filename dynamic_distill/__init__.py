from importlib import metadata


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("dynamic_distill")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)
