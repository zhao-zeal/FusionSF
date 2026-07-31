"""Utility exports loaded lazily so lightweight record tools do not import Lightning."""

from importlib import import_module


_MODULE_EXPORTS = {
    "get_pylogger": ("src.utils.pylogger", "get_pylogger"),
    "enforce_tags": ("src.utils.rich_utils", "enforce_tags"),
    "print_config_tree": ("src.utils.rich_utils", "print_config_tree"),
    "close_loggers": ("src.utils.utils", "close_loggers"),
    "extras": ("src.utils.utils", "extras"),
    "get_metric_value": ("src.utils.utils", "get_metric_value"),
    "instantiate_callbacks": ("src.utils.utils", "instantiate_callbacks"),
    "instantiate_loggers": ("src.utils.utils", "instantiate_loggers"),
    "log_hyperparameters": ("src.utils.utils", "log_hyperparameters"),
    "save_file": ("src.utils.utils", "save_file"),
    "task_wrapper": ("src.utils.utils", "task_wrapper"),
}


def __getattr__(name):
    if name in {"pylogger", "rich_utils"}:
        return import_module(f"src.utils.{name}")
    if name in _MODULE_EXPORTS:
        module, attribute = _MODULE_EXPORTS[name]
        return getattr(import_module(module), attribute)
    raise AttributeError(name)
