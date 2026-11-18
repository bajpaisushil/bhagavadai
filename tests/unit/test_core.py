from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from bhagavatam_ai.config import (
    ConfigError,
    ModelConfig,
    load_data_config,
    load_hardware_config,
    load_model_config,
    load_section,
)
from bhagavatam_ai.logging_setup import JsonFormatter, configure_logging
from bhagavatam_ai.paths import ProjectPaths, ProjectRootNotFoundError, find_project_root
from bhagavatam_ai.utils.hashing import sha256_file, sha256_text
from bhagavatam_ai.utils.io import atomic_write_text, iter_jsonl, read_json, write_json, write_jsonl
from bhagavatam_ai.utils.reports import git_revision, new_run_id, write_report
from bhagavatam_ai.utils.timing import Timer


def test_real_configs_load(repo_root: Path) -> None:
    model = load_model_config(repo_root / "configs/model.yaml")
    assert model.expected.model_class == "Qwen3_5ForCausalLM" and model.dtype.mps == "bfloat16"
    assert "in_proj_qkv" in model.lora_target_modules
    hw = load_hardware_config(repo_root / "configs/hardware.yaml")
    assert hw.training.seq_lens == (256, 512, 1024, 2048) and hw.mps_memory_fraction == 1.0
    assert load_hardware_config(repo_root / "configs/hardware.cpu-smoke.yaml").device == "cpu"
    assert "vedabase.io" in load_data_config(repo_root / "configs/data.yaml").blocked_domains


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            "model: {name_or_path: x, revision: main, expected: {}, lora_target_modules: [q]}",
            "should match pattern",
        ),
        (
            "model: {name_or_path: x, revision: v1.0.0-rc1-tag, expected: {}, lora_target_modules: [q]}",
            "should match pattern",
        ),
        ("model: {name_or_path: x, revision: abcdef12, typo_key: 1}", "Extra inputs"),
        ("other: {}", "no top-level 'model'"),
        ("- just a list", "mapping at the top level"),
        ("model: [unclosed", "Invalid YAML"),
    ],
)
def test_config_errors(tmp_path: Path, yaml_text: str, message: str) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_section(path, "model", ModelConfig)


@pytest.mark.parametrize(
    ("training", "message"),
    [({"seq_lens": []}, "at least 1"), ({"seq_lens": [1]}, "greater than or equal to 2")],
)
def test_hardware_config_rejects_bad_seq_lens(training: dict[str, object], message: str) -> None:
    from pydantic import ValidationError

    from bhagavatam_ai.config import HardwareCheckConfig

    with pytest.raises(ValidationError, match=message):
        HardwareCheckConfig.model_validate({"training": training})


def test_dedup_config_requires_divisible_bands() -> None:
    from pydantic import ValidationError

    from bhagavatam_ai.config import DedupConfig

    with pytest.raises(ValidationError, match="divisible"):
        DedupConfig(num_perm=64, bands=12)


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "nope.yaml")


def test_project_root_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.delenv("BHAGAVATAM_ROOT", raising=False)
    assert find_project_root(nested) == root.resolve()
    paths = ProjectPaths.discover(nested)
    assert paths.raw == root.resolve() / "data" / "raw" and paths.resolve("x/y") == root.resolve() / "x/y"
    assert paths.resolve("/abs") == Path("/abs")
    with pytest.raises(ProjectRootNotFoundError):
        find_project_root(tmp_path / "elsewhere")
    monkeypatch.setenv("BHAGAVATAM_ROOT", str(root))
    assert find_project_root(Path("/")) == root.resolve()
    monkeypatch.setenv("BHAGAVATAM_ROOT", str(tmp_path))
    with pytest.raises(ProjectRootNotFoundError, match="does not look like"):
        find_project_root()


def test_io_roundtrip(tmp_path: Path) -> None:
    write_json(tmp_path / "a/b.json", {"k": "Kṛṣṇa"})
    assert read_json(tmp_path / "a/b.json") == {"k": "Kṛṣṇa"}
    assert "Kṛṣṇa" in (tmp_path / "a/b.json").read_text(encoding="utf-8")  # not \\u-escaped
    assert write_jsonl(tmp_path / "r.jsonl", [{"i": 1}, {"i": 2}]) == 2
    assert [r["i"] for r in iter_jsonl(tmp_path / "r.jsonl")] == [1, 2]
    (tmp_path / "bad.jsonl").write_text('{"ok": 1}\n[1]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        list(iter_jsonl(tmp_path / "bad.jsonl"))
    atomic_write_text(tmp_path / "t.txt", "x")
    assert not list(tmp_path.glob(".t.txt.*"))  # no temp files left behind
    umask = os.umask(0)
    os.umask(umask)
    assert (tmp_path / "t.txt").stat().st_mode & 0o777 == 0o666 & ~umask  # not mkstemp's 0o600


def test_hashing(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"abc")
    assert (
        sha256_file(tmp_path / "f")
        == sha256_text("abc")
        == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    )


def test_reports(tmp_path: Path, repo_root: Path) -> None:
    run_id = new_run_id()
    path = write_report({"x": 1}, reports_dir=tmp_path, name="demo", run_id=run_id, project_root=repo_root)
    doc = read_json(path)
    assert doc["x"] == 1 and doc["_meta"]["run_id"] == run_id and "commit" in doc["_meta"]["git"]
    assert read_json(tmp_path / "demo" / "latest.json") == doc
    assert git_revision(tmp_path) == {"commit": None, "dirty": None}


def test_json_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    configure_logging("INFO", log_file=log_file)
    logging.getLogger("t").info("hello %s", "world", extra={"stage": "parse"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["message"] == "hello world" and entry["stage"] == "parse" and entry["level"] == "INFO"
    configure_logging("WARNING")
    assert isinstance(JsonFormatter().format(logging.LogRecord("n", 20, "p", 1, "m", None, None)), str)


def test_timer() -> None:
    with Timer() as timer:
        pass
    assert timer.elapsed >= 0
