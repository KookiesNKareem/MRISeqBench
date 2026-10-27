import json

from mriseqbench.harness import remote_author


def test_stage_drain_cannot_shutdown_a_host_used_by_another_stage(tmp_path, monkeypatch):
    def forbidden_remote(*args, **kwargs):
        raise AssertionError("A stage drain must not issue a host command")

    monkeypatch.setattr(remote_author, "remote", forbidden_remote)
    host = {"instance_id": "owned-test-host"}
    earlier_stage = tmp_path / "earlier"
    later_stage = tmp_path / "later"
    earlier_stage.mkdir()
    later_stage.mkdir()
    (later_stage / "active-author.json").write_text('{"running": true}')

    remote_author.request_host_drain(host, earlier_stage)

    request = json.loads(
        (earlier_stage / "owned-test-host-drain-request.json").read_text()
    )
    assert request["shutdown_requested"] is False
    assert request["requires_host_lease_owner_verification"] is True
    assert (later_stage / "active-author.json").exists()
    assert not (earlier_stage / "owned-test-host-retirement.json").exists()
