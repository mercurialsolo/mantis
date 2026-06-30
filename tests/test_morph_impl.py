"""Tests for the Morph microVM computer-plane backend — fake SDK, no real API."""

from __future__ import annotations


import pytest


class _FakeMorphInstance:
    def __init__(self, iid: str = "inst-1", host: str = "cua-inst-1.http.morph.so"):
        self.id = iid
        self._host = host
        self.stopped = False
        self.exposed: list[tuple[str, int]] = []

    def expose_http_service(self, name: str, port: int) -> str:
        self.exposed.append((name, port))
        return f"https://{self._host}"

    def stop(self) -> None:
        self.stopped = True


class _FakeInstances:
    def __init__(self, instance: _FakeMorphInstance):
        self._instance = instance
        self.start_calls = 0

    def start(self, snapshot_id: str) -> _FakeMorphInstance:
        self.start_calls += 1
        return self._instance


class _FakeMorphClient:
    """Has ``.instances`` so MorphComputerImpl._make_client returns it as-is."""

    def __init__(self, instance: _FakeMorphInstance):
        self.instances = _FakeInstances(instance)


def _install_fake_requests(monkeypatch, ok: bool = True) -> None:
    import requests

    class _Resp:
        def __init__(self, code: int):
            self.status_code = code

    def _fake_get(url, timeout=None):  # noqa: ANN001
        if ok:
            return _Resp(200)
        raise requests.ConnectionError("simulated connection refused")

    monkeypatch.setattr(requests, "get", _fake_get)


def test_morph_provisions_and_ready(monkeypatch) -> None:
    from mantis_agent.gym.morph_impl import MorphComputerImpl

    _install_fake_requests(monkeypatch, ok=True)
    inst = _FakeMorphInstance()
    client = _FakeMorphClient(inst)
    impl = MorphComputerImpl(
        api_key="x", snapshot_id="snap-1", sdk_module=client,
        startup_timeout_seconds=2.0,
        tenant_id="acme", profile_id="p", run_id="r",
    )
    assert impl._base_url == "https://cua-inst-1.http.morph.so"
    assert client.instances.start_calls == 1
    assert inst.exposed == [("cua", 8000)]
    # provisioned → owns the instance → close() stops it
    impl.close()
    assert inst.stopped is True


def test_morph_claim_mode_skips_provision_and_does_not_stop(monkeypatch) -> None:
    """Claim mode: pre-booted instance + base_url → no SDK, no teardown on close."""
    from mantis_agent.gym.morph_impl import MorphComputerImpl

    inst = _FakeMorphInstance()
    impl = MorphComputerImpl(
        instance=inst, base_url="https://prewarmed.morph.so",
        tenant_id="acme", profile_id="p", run_id="r",
    )
    assert impl._base_url == "https://prewarmed.morph.so"
    impl.close()
    # pool owns claimed instances → MUST NOT be stopped by the run
    assert inst.stopped is False


def test_morph_health_timeout_tears_down(monkeypatch) -> None:
    from mantis_agent.gym.morph_impl import MorphComputerImpl
    import mantis_agent.gym.morph_impl as morph_mod

    _install_fake_requests(monkeypatch, ok=False)
    monkeypatch.setattr(morph_mod.time, "sleep", lambda *a, **kw: None)
    inst = _FakeMorphInstance()
    client = _FakeMorphClient(inst)

    with pytest.raises(TimeoutError):
        MorphComputerImpl(
            api_key="x", snapshot_id="snap-1", sdk_module=client,
            startup_timeout_seconds=0.1,
        )
    # failed boot MUST stop the instance so we don't leak compute
    assert inst.stopped is True


def test_morph_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("MORPH_API_KEY", raising=False)
    from mantis_agent.gym.morph_impl import MorphComputerImpl

    with pytest.raises(ValueError, match="Morph API key"):
        MorphComputerImpl(sdk_module=_FakeMorphClient(_FakeMorphInstance()))


def test_morph_missing_snapshot_raises(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_MORPH_SNAPSHOT", raising=False)
    from mantis_agent.gym.morph_impl import MorphComputerImpl

    with pytest.raises(ValueError, match="snapshot_id"):
        MorphComputerImpl(api_key="x", sdk_module=_FakeMorphClient(_FakeMorphInstance()))
