"""Tests for the Morph pre-warm pool — fake SDK, synchronous refill, no real API."""

from __future__ import annotations

import pytest


class _FakeInst:
    def __init__(self, iid: int):
        self.id = iid
        self.stopped = False

    def expose_http_service(self, name: str, port: int) -> str:
        return f"https://inst-{self.id}.morph.so"

    def stop(self) -> None:
        self.stopped = True


class _FakeInstances:
    def __init__(self) -> None:
        self.start_calls = 0
        self.made: list[_FakeInst] = []

    def start(self, snapshot_id: str) -> _FakeInst:
        self.start_calls += 1
        inst = _FakeInst(self.start_calls)
        self.made.append(inst)
        return inst


class _FakeClient:
    def __init__(self) -> None:
        self.instances = _FakeInstances()


@pytest.fixture
def fake_requests_ok(monkeypatch):
    import requests

    class _Resp:
        status_code = 200

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _Resp())


def _pool(client, size=2):
    from mantis_agent.gym.morph_prewarm_pool import MorphPrewarmPool

    return MorphPrewarmPool(
        api_key="x", snapshot_id="snap-1", size=size,
        sdk_module=client, refill_async=False, startup_timeout_seconds=2.0,
    )


def test_top_up_boots_to_size(fake_requests_ok) -> None:
    client = _FakeClient()
    pool = _pool(client, size=3)
    booted = pool.top_up()
    assert booted == 3
    assert pool.ready_count == 3
    assert client.instances.start_calls == 3
    # idempotent at size
    assert pool.top_up() == 0
    assert client.instances.start_calls == 3


def test_acquire_pops_ready_without_booting(fake_requests_ok) -> None:
    client = _FakeClient()
    pool = _pool(client, size=2)
    pool.top_up()
    assert client.instances.start_calls == 2
    warm = pool.acquire()
    assert warm.base_url.startswith("https://inst-")
    assert pool.ready_count == 1
    # refill_async=False → no auto-reboot on acquire
    assert client.instances.start_calls == 2


def test_acquire_cold_miss_boots(fake_requests_ok) -> None:
    client = _FakeClient()
    pool = _pool(client, size=1)
    # never topped up → empty → acquire boots on demand
    warm = pool.acquire()
    assert warm.instance is not None
    assert client.instances.start_calls == 1


def test_release_stops_instance(fake_requests_ok) -> None:
    client = _FakeClient()
    pool = _pool(client, size=1)
    warm = pool.acquire()  # cold-boot one
    pool.release(warm.instance)
    assert warm.instance.stopped is True


def test_drain_stops_all_ready(fake_requests_ok) -> None:
    client = _FakeClient()
    pool = _pool(client, size=2)
    pool.top_up()
    ready = list(client.instances.made)
    pool.drain()
    assert pool.ready_count == 0
    assert all(i.stopped for i in ready)
    # closed pool refuses acquire
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire()
