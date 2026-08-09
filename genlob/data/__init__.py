"""Data helpers with lazy imports for optional dataframe dependencies."""

from importlib import import_module

__all__ = [
    "derive_lobster_action_labels",
    "lobster_actions_to_array",
    "load_lobster_pair",
    "read_message_csv",
    "read_orderbook_csv",
    "load_nasdaq_multiday",
    "NASDAQMultiDayDataset",
    "FI2010WindowDataset",
    "get_fi2010_loaders",
    "load_fi2010_arrays",
]


_EXPORT_MODULES = {
    "derive_lobster_action_labels": ".lobster_actions",
    "lobster_actions_to_array": ".lobster_actions",
    "load_lobster_pair": ".lobster",
    "read_message_csv": ".lobster",
    "read_orderbook_csv": ".lobster",
    "load_nasdaq_multiday": ".nasdaq",
    "NASDAQMultiDayDataset": ".nasdaq",
    "FI2010WindowDataset": ".fi2010",
    "get_fi2010_loaders": ".fi2010",
    "load_fi2010_arrays": ".fi2010",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
