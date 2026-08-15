import pytest

from solarflare.data.base import BaseCollector


def test_base_collector_is_abstract():
    with pytest.raises(TypeError):
        BaseCollector()