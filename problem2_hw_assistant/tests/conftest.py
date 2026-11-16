import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from hwassistant.agent import Assistant  # noqa: E402
from hwassistant.executor import FixtureExecutor  # noqa: E402
from hwassistant.offline import OfflineBackend  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")


def read(fixture, rel):
    return open(os.path.join(FIX, fixture, rel)).read()


@pytest.fixture()
def bot():
    return Assistant(OfflineBackend(), FixtureExecutor(os.path.join(FIX, "laptop_16gb")))


@pytest.fixture()
def bot_no_root():
    return Assistant(OfflineBackend(), FixtureExecutor(os.path.join(FIX, "desktop_no_root")))
