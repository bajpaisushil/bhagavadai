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


def test_errors_map_to_exit_codes(project: Path) -> None:
    (project / "configs/model.yaml").write_text("model: {}", encoding="utf-8")
    assert main(["model", "download"]) == 1  # invalid config -> error exit, no traceback
    assert main(["data", "parse", "--source", "does-not-exist"]) == 1


def test_hardware_ops_cpu(project: Path) -> None:
    assert main(["hardware", "ops", "--config", "configs/hardware.cpu-smoke.yaml"]) == 0
    report = json.loads((project / "reports/hardware_ops/latest.json").read_text())
    assert report["device"] == "cpu" and all(r["status"] == "pass" for r in report["results"])
