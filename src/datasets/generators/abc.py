from __future__ import annotations

from abc import ABC, abstractmethod

# Shared row shape for generated transactions.
TransactionRow = dict[str, int | float | str]


class Generator(ABC):
    """Interface for stateless or stateful generator components.

    Input:
        Arbitrary positional and keyword arguments, depending on implementation.
    Output:
        Component-specific generated object(s).
    What it does:
        Defines the common `generate(...)` contract for all synthetic generators.
    """

    @abstractmethod
    def generate(self, *args: object, **kwargs: object) -> object:
        """Generate data according to the concrete implementation.

        Input:
            *args/**kwargs: implementation-specific generation parameters.
        Output:
            Generated value(s), shape determined by implementation.
        What it does:
            Forces subclasses to implement generation behavior.
        """
        raise NotImplementedError


class Build(ABC):
    """Interface for objects that build a final artifact (for example a dataset)."""

    @abstractmethod
    def build(self, *args: object, **kwargs: object) -> TransactionRow:
        """Build the final artifact for the concrete implementation.

        Input:
            *args/**kwargs: implementation-specific build parameters.
        Output:
            Built artifact (concrete return type may be narrower than alias).
        What it does:
            Forces subclasses to provide a deterministic build entry point.
        """
        raise NotImplementedError
