"""Keep every test out of the real state directory.

Found the hard way: the disclosure ledger writes on each outbound request, the
fake clients in these tests make requests to a path called "x", and
state_dir() resolves to /var/lib/faethon whenever it exists and is readable.
So a full test run silently appended forty rows to the live privacy ledger --
the one file in this project whose entire value is being an accurate record of
what actually happened.

An autouse fixture rather than a per-test one, because remembering to isolate
is exactly the thing nobody does. The timers and the turn log write there too
and had the same exposure.
"""

from __future__ import annotations

import pytest

from faethon import state


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "state_dir", lambda: tmp_path)
    return tmp_path
