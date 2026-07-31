"""Metrics package with lazy optional torchmetrics imports."""


def __getattr__(name):
    if name == "MeanAbsolutePercentageError":
        from .mape import MeanAbsolutePercentageError
        return MeanAbsolutePercentageError
    raise AttributeError(name)
