"""Public ITCH snapshot data helpers."""

from importlib import import_module

__all__ = [
    "load_nasdaq_multiday",
    "NASDAQMultiDayDataset",
]


_EXPORT_MODULES = {
    "load_nasdaq_multiday": ".nasdaq",
    "NASDAQMultiDayDataset": ".nasdaq",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
