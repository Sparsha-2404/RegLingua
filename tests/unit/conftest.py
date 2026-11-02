"""
DAY 13 (M2): offline-safe test policy for tests/unit.

  * marker  hf_network : the test really needs the Hugging Face Hub / network. It is SKIPPED (reason shown by
                         `pytest -rs`) when offline: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / RL_OFFLINE is set, or
                         huggingface.co is unreachable. Mark such tests with `@pytest.mark.hf_network`.
  * safety net         : an UNMARKED test that fails with a recognised network / HF-offline error is reported as
                         SKIPPED ("network unavailable: ...") instead of FAILED, and listed at the end of the run so the
                         marker can be added. Ordinary assertion failures and missing-package errors are never masked.
The marker is registered here (not in pytest.ini, which this file does not own).
"""
from __future__ import annotations

import os
import socket

import pytest

_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "RL_OFFLINE")
_TRUE = {"1", "true", "yes", "on"}
_NET_SIGNATURES = (
    "couldn't connect to 'https://huggingface.co'", "we couldn't connect", "localentrynotfounderror",
    "maxretryerror", "name resolution", "nodename nor servname", "network is unreachable",
    "connection refused", "connection reset", "connection aborted", "timed out",
    "outgoing traffic has been disabled", "offline mode is enabled", "cannot find the requested files in the local cache",
    "hf_hub_offline", "failed to resolve", "getaddrinfo failed",
)
_cache: dict = {}
_auto_skipped: list = []


def offline_reason() -> str:
    """'' when the network looks usable, otherwise a short reason (cached)."""
    if "r" in _cache:
        return _cache["r"]
    for k in _OFFLINE_ENV:
        if os.environ.get(k, "").strip().lower() in _TRUE:
            _cache["r"] = f"offline: {k}={os.environ[k]}"
            return _cache["r"]
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        _cache["r"] = ""
    except OSError as exc:
        _cache["r"] = f"offline: huggingface.co unreachable ({type(exc).__name__})"
    return _cache["r"]


def is_network_error(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in _NET_SIGNATURES)


def pytest_configure(config):
    config.addinivalue_line("markers", "hf_network: needs Hugging Face Hub / network access; skipped when offline")


def pytest_collection_modifyitems(config, items):
    marked = [i for i in items if i.get_closest_marker("hf_network")]
    if not marked:
        return
    why = offline_reason()
    if why:
        skip = pytest.mark.skip(reason=f"needs Hugging Face/network ({why}); run once online or unset HF_HUB_OFFLINE")
        for i in marked:
            i.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call" or not rep.failed or call.excinfo is None:
        return
    if call.excinfo.errisinstance((AssertionError, ImportError)):   # real failures / missing packages stay failures
        return
    msg = f"{call.excinfo.typename}: {call.excinfo.value}"
    if is_network_error(msg) and offline_reason():
        reason = f"network unavailable: {msg[:140]}"
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {reason}")
        _auto_skipped.append(item.nodeid)


def pytest_terminal_summary(terminalreporter):
    if _auto_skipped:
        terminalreporter.write_sep("-", "unmarked tests skipped on a network error (add @pytest.mark.hf_network)")
        for n in _auto_skipped:
            terminalreporter.write_line(n)
