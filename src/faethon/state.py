"""The first thing Faethon writes down.

Everything else is proudly ephemeral -- memory forgets on restart, forgets
again after ten minutes of quiet, and nothing has ever survived the process.
Timers are the first thing that must, because a pasta timer lost to a restart
is a burnt dinner rather than a forgotten conversation.

Lives in systemd's StateDirectory (/var/lib/faethon), which the unit declares:
systemd creates it, owns it to the service user, and adds it to the sandbox's
writable paths, so none of ProtectHome or ProtectSystem has to be relaxed. Away
from systemd -- tests, a bare `uv run faethon` -- it falls back to a directory
beside the code.

Writes are atomic. A crash between opening the file and finishing the write
would otherwise leave a truncated JSON document that fails to parse on the next
boot, turning a lost timer into a permanently broken one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)


def state_dir() -> Path:
    """Where persistent state goes, creating it if needed."""
    env = os.environ.get("STATE_DIRECTORY")
    path = Path(env.split(":")[0]) if env else PROJECT_ROOT / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load(name: str, default: Any) -> Any:
    """Read `name`.json, or return `default` if it is absent or unreadable.

    A corrupt file is a warning and a fresh start, never an exception: state
    that cannot be parsed must not stop the assistant from running.
    """
    path = state_dir() / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring unreadable state at %s: %s", path, e)
        return default


def save(name: str, data: Any) -> None:
    """Write `name`.json atomically. Failures are logged, never raised."""
    path = state_dir() / f"{name}.json"
    try:
        # Same directory, so the rename is atomic rather than a cross-device
        # copy that could be interrupted half-written.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as e:
        log.error("could not save state to %s: %s", path, e)
