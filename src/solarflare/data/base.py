from abc import ABC, abstractmethod


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    @abstractmethod
    def collect(self):
        """Download data."""
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """Validate downloaded data."""
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """Persist data."""
        raise NotImplementedError