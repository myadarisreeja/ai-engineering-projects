import platform

import pytest

from hwassistant.agent import Assistant
from hwassistant.executor import SystemExecutor
from hwassistant.offline import OfflineBackend

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")


def test_real_cpu_and_storage_smoke():
    bot = Assistant(OfflineBackend(), SystemExecutor())
    assert "CPU" in bot.ask("what cpu do I have")
    assert bot.ask("how much ram do I have?")
    assert "can't determine" in bot.ask("who won the cricket match").lower()
