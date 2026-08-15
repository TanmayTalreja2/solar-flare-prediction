from solarflare.data.base import BaseCollector


class SHARPCollector(BaseCollector):
    """Collector responsible for SHARP observations."""

    def collect(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError