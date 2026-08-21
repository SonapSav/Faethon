"""How Faethon itself is doing: temperature, throttling, and the network.

One skill rather than five, because Registry.match is first-match-wins in
import order -- so separate temperature, IP, SSID and status skills would
compete over "how are you", "what's your status", "what's your IP", and which
one won would depend on filenames. The combined answer is also the one anyone
actually wants: temperature, link and address in a sentence.

The half that earns its keep is tick(), not run(). A CPU temperature you have
to ask for is nearly useless, because you would never think to ask -- you would
ask a week after the wake word started failing. Throttling degrades Faethon
specifically: openWakeWord runs continuously, a Pi 4 soft-throttles at 80C, and
the symptom is missed or late wake words.

Under-voltage is the one worth waiting for. A marginal supply or a mediocre
cable causes intermittent throttling that looks exactly like a software bug,
and vcgencmd latches it since boot -- so it can be reported long after the
moment that caused it, which is the only way anyone would ever catch it.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import time
from pathlib import Path

from .base import Skill

log = logging.getLogger(__name__)

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
#: vcgencmd forks a process, so this is not something to do per audio frame.
CHECK_EVERY_SEC = 60.0

#: get_throttled bits. The low ones are now; the high ones latched since boot.
UNDER_VOLTAGE_NOW = 1 << 0
THROTTLED_NOW = 1 << 2
UNDER_VOLTAGE_EVER = 1 << 16
THROTTLED_EVER = 1 << 18

#: A Pi 4 soft-throttles at 80 and hard-throttles at 85. A bare number is
#: alarming to anyone who does not know that and reassuring to anyone who
#: does, so the reply carries the judgement, as volume_skill does with dB.
_BANDS = [(60, "normal"), (70, "warm"), (80, "hot"), (200, "throttling")]


def _run(*cmd: str) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_temp() -> float | None:
    try:
        return int(THERMAL.read_text().strip()) / 1000
    except (OSError, ValueError):
        return None


def describe_temp(celsius: float) -> str:
    for limit, word in _BANDS:
        if celsius < limit:
            return word
    return "throttling"


def throttle_flags() -> int | None:
    """The get_throttled bitmask, or None where vcgencmd isn't available."""
    out = _run("vcgencmd", "get_throttled")
    m = re.search(r"0x([0-9a-fA-F]+)", out)
    return int(m.group(1), 16) if m else None


def link() -> tuple[str, str, str]:
    """(interface, address, ssid). Empty strings where unknown."""
    route = _run("ip", "route", "get", "1.1.1.1")
    iface = (re.search(r"dev (\S+)", route) or [None, ""])[1]
    address = (re.search(r"src (\S+)", route) or [None, ""])[1]
    if not iface:
        return "", address, ""
    # Wireless interfaces have this directory and wired ones do not. Cheaper
    # and more durable than matching on names, which predictable naming turns
    # into things like wlp2s0.
    if not Path(f"/sys/class/net/{iface}/wireless").exists():
        return iface, address, ""
    # A ladder rather than one command: which of these exists depends on the
    # Pi OS generation, NetworkManager having replaced wpa_supplicant.
    ssid = ""
    for cmd in (("iwgetid", "-r"), ("iw", "dev", iface, "link"),
                ("nmcli", "-t", "-f", "active,ssid", "dev", "wifi")):
        out = _run(*cmd)
        if not out:
            continue
        if cmd[0] == "iwgetid":
            ssid = out.splitlines()[0]
        elif cmd[0] == "iw":
            m = re.search(r"SSID:\s*(.+)", out)
            ssid = m.group(1).strip() if m else ""
        else:
            for row in out.splitlines():
                if row.startswith("yes:"):
                    ssid = row.split(":", 1)[1]
        if ssid:
            break
    return iface, address, ssid


def spoken_address(address: str) -> str:
    """An IP the way someone could actually write it down.

    Auditioned through the speaker: the raw string and a dot-separated one both
    came out around 5.1s, and digits grouped by octet at 4.3s -- shorter and
    far easier to transcribe by ear, which is the only reason anyone asks.
    """
    parts = address.split(".")
    if len(parts) != 4:
        return address
    # Three-digit octets are spelled out, shorter ones left as numbers:
    # "one hundred" is ambiguous where "1 0 0" is not, and "61" is easier to
    # catch than "6 1". Grouped by octet so the dots are audible as pauses.
    return ", ".join(" ".join(o) if len(o) == 3 else o for o in parts)


class HealthSkill(Skill):
    name = "get_health"
    tag = "utility"
    description = (
        "Report how the Raspberry Pi itself is doing: CPU temperature, whether "
        "it is throttling, which network it is on and at what address."
    )

    _END = r"(?:\s+(?:now|please|then))*[^\w]*$"
    patterns = [
        rf"\bhow are you (?:doing|feeling){_END}",
        rf"\bare you (?P<status>ok|okay|alright|healthy){_END}",
        rf"\bwhat(?:'s|s| is) your status{_END}",
        r"\bhow (?P<temp>hot|warm) are you\b",
        r"\bwhat(?:'s|s| is) your (?P<temp>temperature|cpu temperature)\b",
        r"\bare you (?P<temp>overheating|throttling|throttled)\b",
        r"\bwhat(?:'s|s| is) (?:your|my|the) (?P<addr>i\.?p\.?(?: address)?)\b",
        r"\bwhat (?P<addr>address) are you on\b",
        r"\bwhat (?P<net>network|wifi|wi-fi) (?:are you|am i|is it) on\b",
        r"\bare you on (?P<net>wifi|wi-fi|ethernet)\b",
        r"\bwhat(?:'s|s| is) (?:the )?(?P<net>ssid|wifi name)\b",
        # Asked "are you connected to a wifi network", the model answered with
        # a specific SSID without calling anything.
        r"\bare you (?P<net>connected|online)\b",
        r"\bis the (?P<net>wifi|wi-fi|network|internet) (?:working|up|on|ok|okay|down)\b",
        r"\bhow(?:'s|s| is) your (?P<temp>cpu|processor)\b",
    ]

    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["status", "temperature", "address", "network"],
                "description": "Defaults to a summary of all of it.",
            }
        },
        "required": [],
    }

    def __init__(self) -> None:
        super().__init__()
        # Far enough back that the first tick checks. Starting at 0 meant the
        # first check was always "too soon" and nothing was looked at for the
        # first minute of every run.
        self._last_check = -CHECK_EVERY_SEC
        self._warned: set[str] = set()

    # -- the unprompted half ----------------------------------------------

    def tick(self) -> str | None:
        """Speak up once about throttling or a marginal power supply."""
        now = time.monotonic()
        if now - self._last_check < CHECK_EVERY_SEC:
            return None
        self._last_check = now

        flags = throttle_flags()
        if flags is None:
            return None
        if flags & (UNDER_VOLTAGE_NOW | UNDER_VOLTAGE_EVER) and "power" not in self._warned:
            self._warned.add("power")
            log.warning("under-voltage flag set (0x%x)", flags)
            return (
                "I've had under-voltage warnings, which usually means the "
                "power supply or the cable. It can make me unreliable."
            )
        if flags & (THROTTLED_NOW | THROTTLED_EVER) and "heat" not in self._warned:
            self._warned.add("heat")
            temp = cpu_temp()
            log.warning("throttling flag set (0x%x), %s C", flags, temp)
            return (
                "I've been throttling from heat, which can make me slow to "
                "hear you. I could do with better airflow."
            )
        return None

    # -- the asked-for half -----------------------------------------------

    def run(self, **params: object) -> str:
        kind = str(params.get("kind") or "").lower()
        if not kind:
            if "temp" in params:
                kind = "temperature"
            elif "addr" in params:
                kind = "address"
            elif "net" in params:
                kind = "network"
            else:
                kind = "status"

        if kind == "temperature":
            return self._temperature()
        if kind == "address":
            return self._address()
        if kind == "network":
            return self._network()
        return f"{self._temperature()} {self._network()}"

    def _temperature(self) -> str:
        c = cpu_temp()
        if c is None:
            return "I can't read my temperature."
        said = f"I'm at {round(c)} degrees, which is {describe_temp(c)}."
        flags = throttle_flags()
        if flags and flags & (UNDER_VOLTAGE_NOW | UNDER_VOLTAGE_EVER):
            said += " I've also had under-voltage warnings."
        return said

    def _address(self) -> str:
        _, address, _ = link()
        host = f"{socket.gethostname()} dot local"
        if not address:
            return f"I don't have an address right now. Try {host}."
        # Hostname first: it is stable, half the length spoken, and the only
        # one anyone can reliably write down by ear.
        return f"You can reach me at {host}, or at {spoken_address(address)}."

    def _network(self) -> str:
        iface, address, ssid = link()
        if not iface:
            return "I don't seem to be on a network."
        if not ssid:
            return "I'm on ethernet."
        return f"I'm on wi-fi, connected to {ssid}."


SKILL = HealthSkill()
