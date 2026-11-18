from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bhagavatam_ai.cli import build_parser, main


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and "bhagavatam" in capsys.readouterr().out
    parser = build_parser()
    for argv in (["env"], ["model", "inspect"], ["hardware", "check", "--skip-training"], ["data", "build"]):
        assert callable(parser.parse_args(argv).func)


def test_env_writes_report(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    latest = project / "reports/environment/latest.json"
    assert json.loads(latest.read_text())["environment"]["os"]
    assert "report:" in capsys.readouterr().out


def test_data_commands_on_fixture(
    project: Path, repo_root: Path, fixtures_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    manifest = yaml.safe_load((repo_root / "data/manifests/sources.yaml").read_text(encoding="utf-8"))
    gretil = next(s for s in manifest["sources"] if s["id"] == "gretil-bhagavata-purana")
    gretil["downloads"][0]["sha256"] = None  # the fixture stands in for the real file
    manifest["sources"] = [gretil]
    (project / "data/manifests/sources.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    raw = project / "data/raw/gretil-bhagavata-purana"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "mini_gretil.xml", raw / "sa_bhAgavatapurANa.xml")

    assert main(["data", "manifest"]) == 0
    assert main(["data", "build"]) == 0
    out = capsys.readouterr().out
    assert "gretil-bhagavata-purana: ok=True" in out and "No source is cleared" in out
    assert (project / "data/reference/expected_structure.json").exists()
    assert main(["data", "validate"]) == 0
    assert main(["data", "reference", "--source", "gretil-bhagavata-purana"]) == 0
    assert "wrote data/reference/expected_structure.json" in capsys.readouterr().out


def _write_gretil_manifest(project: Path, repo_root: Path) -> None:
    import yaml

    manifest = yaml.safe_load((repo_root / "data/manifests/sources.yaml").read_text(encoding="utf-8"))
    manifest["sources"] = [s for s in manifest["sources"] if s["id"] == "gretil-bhagavata-purana"]
    manifest["sources"][0]["downloads"][0]["sha256"] = None
    text = yaml.safe_dump(manifest, allow_unicode=True)
    (project / "data/manifests/sources.yaml").write_text(text, encoding="utf-8")


def test_errors_map_to_exit_codes(project: Path, repo_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_gretil_manifest(project, repo_root)
    assert main(["data", "parse", "--source", "does-not-exist"]) == 1
    assert "Unknown source id 'does-not-exist'" in caplog.text  # fails for the right reason
    (project / "configs/model.yaml").write_text("model: {}", encoding="utf-8")
    assert main(["model", "download"]) == 1  # invalid config -> error exit, no traceback


def test_validate_does_not_pass_silently(
    project: Path, repo_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_gretil_manifest(project, repo_root)
    assert main(["data", "validate", "--source", "gretil-bhagavata-purana"]) == 1  # not parsed yet
    assert "run the parse stage for gretil-bhagavata-purana first" in caplog.text
    assert str(project) not in caplog.text  # paths are shown relative to the project
    assert main(["data", "validate"]) == 1  # nothing parsed at all
    assert "nothing to validate" in caplog.text


def test_bad_log_file_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--log-file", str(tmp_path), "env"]) == 1  # a directory, not a file
    assert "cannot open log file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("verdict", "strict", "code"),
    [
        ({"checks_ok": False, "ready_for_phase2": False}, False, 2),
        ({"checks_ok": True, "ready_for_phase2": False}, False, 0),
        ({"checks_ok": True, "ready_for_phase2": False}, True, 2),
        ({"checks_ok": True, "ready_for_phase2": True}, True, 0),
    ],
)
def test_hardware_check_exit_codes(
    project: Path, monkeypatch: pytest.MonkeyPatch, verdict: dict[str, bool], strict: bool, code: int
) -> None:
    from bhagavatam_ai.hardware import benchmark

    monkeypatch.setattr(benchmark, "run_hardware_check", lambda *a, **k: {"verdict": verdict, "training": {}})
    assert main(["hardware", "check", *(["--strict"] if strict else [])]) == code


def test_hardware_ops_cpu(project: Path) -> None:
    assert main(["hardware", "ops", "--config", "configs/hardware.cpu-smoke.yaml"]) == 0
    report = json.loads((project / "reports/hardware_ops/latest.json").read_text())
    assert report["device"] == "cpu" and all(r["status"] == "pass" for r in report["results"])
