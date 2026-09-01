try:
    import importlib.metadata as im

    __version__ = im.version(__package__)
except im.PackageNotFoundError:
    __version__ = "0.0.1"

from . import combinatorial, dqi, integrations, opt_class
