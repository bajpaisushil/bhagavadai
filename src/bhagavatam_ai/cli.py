"""``bhagavatam`` command-line interface.

Every command writes a JSON report under ``reports/<name>/<run_id>.json`` (plus ``latest.json``) and
prints a short summary. Exit codes: 0 success, 1 error, 2 checks ran but failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bhagavatam_ai import __version__
from bhagavatam_ai.config import (
    ConfigError,
    load_data_config,
    load_hardware_config,
    load_model_config,
)
from bhagavatam_ai.logging_setup import configure_logging
from bhagavatam_ai.paths import ProjectPaths, ProjectRootNotFoundError
from bhagavatam_ai.utils.reports import new_run_id, write_report

logger = logging.getLogger("bhagavatam_ai.cli")
EXIT_OK, EXIT_ERROR, EXIT_CHECKS_FAILED = 0, 1, 2

CONTROL_TOKEN_IDS = {
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|endoftext|>": 248044,
    "<think>": 248068,
    "</think>": 248069,
    "<|fim_pad|>": 248063,
}


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.paths = ProjectPaths.discover()
        self.run_id = new_run_id()
        self.reports_dir = self.paths.resolve(args.reports_dir) if args.reports_dir else self.paths.reports

    def config_path(self, name: str) -> Path:
        override = getattr(self.args, "config", None)
        return self.paths.resolve(override) if override else self.paths.configs / name

    def report(self, name: str, payload: dict[str, Any]) -> Path:
        path = write_report(
            payload, reports_dir=self.reports_dir, name=name, run_id=self.run_id, project_root=self.paths.root
        )
        print(
            f"report: {path.relative_to(self.paths.root) if path.is_relative_to(self.paths.root) else path}"
        )
        return path


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# --------------------------------------------------------------------------- env / model


def cmd_env(rt: Runtime) -> int:
    from bhagavatam_ai.hardware.device import collect_environment

    env = collect_environment()
    rt.report("environment", {"environment": env})
    _print(env)
    return EXIT_OK


def cmd_model_download(rt: Runtime) -> int:
    from bhagavatam_ai.modeling.loader import download_and_verify

    cfg = load_model_config(rt.paths.configs / "model.yaml")
    result = download_and_verify(cfg)
    rt.report("model_download", result)
    files, size = len(result["files"]), result["total_size_gb"]
    sha, blob = result["verified_sha256"], result["verified_git_blob"]
    print(f"{cfg.name_or_path}@{cfg.revision[:10]}: {files} files, {size} GB")
    print(f"verified: {sha} by SHA-256 (LFS), {blob} by git blob id")
    return EXIT_OK


def cmd_model_inspect(rt: Runtime) -> int:
    from bhagavatam_ai.modeling.loader import load_tokenizer, snapshot_dir
    from bhagavatam_ai.modeling.tokenizer_inspect import embedding_row_norms, inspect_tokenizer

    cfg = load_model_config(rt.paths.configs / "model.yaml")
    snapshot = snapshot_dir(cfg, local_files_only=rt.args.offline)
    tokenizer = load_tokenizer(cfg, snapshot=snapshot)
    tok_report = inspect_tokenizer(tokenizer)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    embed_key = cfg.expected.checkpoint_key_prefix + "embed_tokens.weight"
    norms = embedding_row_norms(snapshot, embed_key, CONTROL_TOKEN_IDS)
    payload = {
        "model": cfg.name_or_path,
        "revision": cfg.revision,
        "architecture": {
            "architectures": config.get("architectures"),
            "text_model_type": text_cfg.get("model_type"),
            "layer_types": {
                t: text_cfg.get("layer_types", []).count(t) for t in set(text_cfg.get("layer_types", []))
            },
            **{
                k: text_cfg.get(k)
                for k in (
                    "hidden_size",
                    "num_hidden_layers",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "linear_num_key_heads",
                    "linear_num_value_heads",
                    "linear_key_head_dim",
                    "linear_value_head_dim",
                    "vocab_size",
                    "max_position_embeddings",
                    "tie_word_embeddings",
                    "mtp_num_hidden_layers",
                )
            },
            "has_vision_tower": "vision_config" in config,
        },
        "tokenizer": tok_report,
        "control_token_embeddings": norms,
    }
    rt.report("model_inspect", payload)
    print(
        f"tokenizer: {tok_report['tokenizer_class']}, vocab {tok_report['vocab_size_with_added']}, "
        f"control tokens single: {tok_report['control_tokens_all_single']}, "
        f"IAST chars/token {tok_report['iast']['mean_chars_per_token']}, "
        f"English tokens/word {tok_report['english']['tokens_per_word']}, "
        f"NFC==NFD tokenization: {tok_report['nfc_and_nfd_tokenize_identically']}"
    )
    print("generation prompt suffix:", repr(tok_report["chat_template"]["generation_prompt_suffix"]))
    return EXIT_OK


def cmd_model_smoke(rt: Runtime) -> int:
    from bhagavatam_ai.hardware.device import dtype_for_device, select_device
    from bhagavatam_ai.hardware.memory import MemoryMonitor
    from bhagavatam_ai.inference.smoke import run_smoke
    from bhagavatam_ai.modeling.loader import load_model, load_tokenizer, snapshot_dir

    cfg = load_model_config(rt.paths.configs / "model.yaml")
    device = select_device(rt.args.device)
    snapshot = snapshot_dir(cfg, local_files_only=rt.args.offline)
    with MemoryMonitor(device) as monitor:
        model, load_report = load_model(cfg, device, dtype_for_device(device, cfg.dtype), snapshot=snapshot)
        tokenizer = load_tokenizer(cfg, snapshot=snapshot)
        result = run_smoke(model, tokenizer, max_new_tokens=rt.args.max_new_tokens)
        monitor.sample_device()
    payload = {"load": load_report.to_dict(), "smoke": result, "memory": monitor.summary()}
    rt.report("model_smoke", payload)
    for item in result["results"]:
        print(
            f"--- [{item['id']}] {item['input']}\n{item['text']}\n"
            f"({item['new_tokens']} tokens, {item['tokens_per_s']} tok/s, "
            f"stopped_on_stop_token={item['stopped_on_stop_token']})"
        )
    return EXIT_OK if result["all_ok"] else EXIT_CHECKS_FAILED


# --------------------------------------------------------------------------- hardware


def cmd_hardware_ops(rt: Runtime) -> int:
    from bhagavatam_ai.hardware.device import dtype_from_name, select_device
    from bhagavatam_ai.hardware.ops_check import descriptions, run_op_checks

    cfg = load_hardware_config(rt.config_path("hardware.yaml"))
    device = select_device(rt.args.device or cfg.device)
    results = run_op_checks(device, [dtype_from_name(d) for d in cfg.op_checks.dtypes], seed=cfg.seed)
    payload = {"device": device.type, "results": [vars(r) for r in results], "descriptions": descriptions()}
    rt.report("hardware_ops", payload)
    for r in results:
        err = "-" if r.max_rel_error is None else f"{r.max_rel_error:.2e}"
        print(f"{r.name:28s} {r.dtype:9s} {r.status:17s} err={err} {r.detail or ''}")
    return EXIT_OK if all(r.status == "pass" for r in results if r.dtype != "float16") else EXIT_CHECKS_FAILED


def cmd_hardware_check(rt: Runtime) -> int:
    from bhagavatam_ai.hardware.benchmark import run_hardware_check

    model_cfg = load_model_config(rt.paths.configs / "model.yaml")
    cfg = load_hardware_config(rt.config_path("hardware.yaml"))
    overrides: dict[str, Any] = {}
    if rt.args.device:
        overrides["device"] = rt.args.device
    if rt.args.sustained_minutes is not None:
        overrides["training"] = cfg.training.model_copy(
            update={"sustained_minutes": rt.args.sustained_minutes}
        )
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    report = run_hardware_check(model_cfg, cfg, skip_training=rt.args.skip_training)
    rt.report("hardware_check", report)
    verdict = report["verdict"]
    _print(verdict)
    training = report.get("training")
    sweep = training.get("sweep", []) if isinstance(training, dict) else []
    for row in sweep:
        mem = row.get("memory", {})
        print(
            f"  seq_len={row['seq_len']:5d} {row['status']:6s} tok/s={row.get('tokens_per_s')} "
            f"peak_driver_gb={mem.get('peak_device_driver_gb')} peak_rss_gb={mem.get('peak_process_rss_gb')} "
            f"pressure={row.get('memory_pressure')}"
        )
    if not verdict["checks_ok"]:
        return EXIT_CHECKS_FAILED
    if rt.args.strict and not verdict["ready_for_phase2"]:
        return EXIT_CHECKS_FAILED
    return EXIT_OK


# --------------------------------------------------------------------------- data


def _data_context(rt: Runtime) -> Any:
    from bhagavatam_ai.data.pipeline import PipelineContext

    return PipelineContext.load(rt.paths, load_data_config(rt.config_path("data.yaml")))


def _selected_sources(
    ctx: Any, requested: Sequence[str] | None, predicate: Callable[[Any], bool]
) -> list[Any]:
    from bhagavatam_ai.data.manifest import get_source

    if requested:
        return [get_source(ctx.manifest, sid) for sid in requested]
    return [s for s in ctx.manifest.sources if predicate(s)]


def cmd_data_manifest(rt: Runtime) -> int:
    ctx = _data_context(rt)
    rows = [
        {
            "id": s.id,
            "status": s.permission_status.value,
            "license": s.license,
            "usage": {k: v for k, v in s.usage.model_dump().items() if v},
            "review": s.review_status,
        }
        for s in ctx.manifest.sources
    ]
    rt.report("data_manifest", {"sources": rows, "updated": str(ctx.manifest.updated)})
    for row in rows:
        print(f"{row['id']:32s} {row['status']:20s} usage={sorted(row['usage']) or '-'}")
    return EXIT_OK


def cmd_data_download(rt: Runtime) -> int:
    from bhagavatam_ai.data.download import download_source

    ctx = _data_context(rt)
    sources = _selected_sources(ctx, rt.args.source, lambda s: s.usage.download)
    results = []
    for entry in sources:
        for r in download_source(
            entry, ctx.paths.raw, ctx.config.download, ctx.config.blocked_domains, force=rt.args.force
        ):
            results.append(vars(r))
            print(f"{r.source_id}: {r.file} {r.status} ({r.bytes:,} bytes, sha256 {r.sha256[:12]}...)")
    rt.report("data_download", {"files": results})
    return EXIT_OK


def cmd_data_parse(rt: Runtime) -> int:
    from bhagavatam_ai.data.pipeline import parse_source

    ctx = _data_context(rt)
    sources = _selected_sources(
        ctx, rt.args.source, lambda s: bool(s.parser) and (s.usage.metadata or s.usage.training_text)
    )
    stats = [parse_source(ctx, s.id) for s in sources]
    rt.report("data_parse", {"sources": stats})
    for st in stats:
        print(
            f"{st['source_id']}: {st['records']} records, {st['chapters']} chapters "
            f"(repaired labels {st.get('labels_repaired')}, "
            f"removed duplicate lines {st.get('duplicate_lines_removed')})"
        )
    return EXIT_OK


def cmd_data_validate(rt: Runtime) -> int:
    from bhagavatam_ai.data.pipeline import validate_source
    from bhagavatam_ai.data.validate import Severity

    ctx = _data_context(rt)
    if rt.args.source:  # explicit ids: a missing parse is an error, not something to skip
        sources = _selected_sources(ctx, rt.args.source, lambda s: True)
    else:
        sources = [s for s in ctx.manifest.sources if s.parser and ctx.interim_file(s.id).exists()]
        if not sources:
            logger.error("nothing to validate: no source has been parsed yet (run `bhagavatam data parse`)")
            return EXIT_ERROR
    reports = [validate_source(ctx, s.id, use_reference=not rt.args.no_reference) for s in sources]
    rt.report("data_validation", {"sources": [r.to_dict() for r in reports]})
    for r in reports:
        counts = {sev.value: r.count(sev) for sev in Severity}
        print(f"{r.source_id}: {r.records} records, ok={r.ok}, issues={counts}")
    return EXIT_OK if all(r.ok for r in reports) else EXIT_CHECKS_FAILED


def cmd_data_reference(rt: Runtime) -> int:
    from bhagavatam_ai.data.pipeline import write_reference_structure

    ctx = _data_context(rt)
    path = write_reference_structure(ctx, rt.args.source)
    shown = path.relative_to(rt.paths.root) if path.is_relative_to(rt.paths.root) else path
    rt.report("data_reference", {"source_id": rt.args.source, "path": str(shown)})
    print(f"wrote {shown}")
    return EXIT_OK


def cmd_data_build(rt: Runtime) -> int:
    """parse every parseable source -> reference structure -> validate -> corpus."""
    from bhagavatam_ai.data.pipeline import (
        build_corpus,
        parse_source,
        validate_source,
        write_reference_structure,
    )

    ctx = _data_context(rt)
    parseable = [s for s in ctx.manifest.sources if s.parser and (s.usage.metadata or s.usage.training_text)]
    parsed = [parse_source(ctx, s.id) for s in parseable]
    reference = next((s for s in parseable if s.reference_structure), None)
    if reference is not None:
        write_reference_structure(ctx, reference.id)
    validations = [validate_source(ctx, s.id).to_dict() for s in parseable]
    card = build_corpus(ctx)
    rt.report("data_build", {"parsed": parsed, "validation": validations, "dataset_card": card})
    for v in validations:
        print(f"{v['source_id']}: ok={v['ok']} counts={v['counts']}")
    print(f"corpus records: {card['records']}" + (f" - {card['note']}" if card.get("note") else ""))
    return EXIT_OK if all(v["ok"] for v in validations) else EXIT_CHECKS_FAILED


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bhagavatam", description="Bhagavatam AI command-line tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-json", action="store_true", help="JSON log lines on stderr")
    parser.add_argument("--log-file", type=Path, help="also write JSON logs to this file")
    parser.add_argument("--reports-dir", help="where reports are written (default: reports/)")
    groups = parser.add_subparsers(dest="group", required=True)

    env = groups.add_parser("env", help="describe OS, hardware and library versions")
    env.set_defaults(func=cmd_env)

    model = groups.add_parser("model", help="download / inspect / smoke-test the base model").add_subparsers(
        dest="command", required=True
    )
    model.add_parser("download", help="download the pinned snapshot and verify checksums").set_defaults(
        func=cmd_model_download
    )
    inspect_p = model.add_parser("inspect", help="tokenizer, chat template and architecture report")
    inspect_p.add_argument("--offline", action="store_true", help="use the local cache only")
    inspect_p.set_defaults(func=cmd_model_inspect)
    smoke = model.add_parser("smoke", help="small end-to-end inference test")
    smoke.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    smoke.add_argument("--max-new-tokens", type=int, default=64)
    smoke.add_argument("--offline", action="store_true")
    smoke.set_defaults(func=cmd_model_smoke)

    hw = groups.add_parser("hardware", help="hardware compatibility and performance checks").add_subparsers(
        dest="command", required=True
    )
    ops = hw.add_parser("ops", help="op-level backend compatibility checks (fast)")
    ops.add_argument("--config", help="hardware config (default configs/hardware.yaml)")
    ops.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"])
    ops.set_defaults(func=cmd_hardware_ops)
    check = hw.add_parser(
        "check", help="full check: ops, load, numerics, generation, LoRA training memory/speed"
    )
    check.add_argument("--config", help="hardware config (default configs/hardware.yaml)")
    check.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"])
    check.add_argument("--skip-training", action="store_true")
    check.add_argument("--sustained-minutes", type=float, help="add a sustained run to measure throttling")
    check.add_argument(
        "--strict",
        action="store_true",
        help="also exit 2 unless ready_for_phase2 (needs an accelerator); failed checks always exit 2",
    )
    check.set_defaults(func=cmd_hardware_check)

    data = groups.add_parser("data", help="licence-gated dataset pipeline").add_subparsers(
        dest="command", required=True
    )
    for name, func, helptext in (
        ("manifest", cmd_data_manifest, "validate and summarise the source/licence manifest"),
        ("download", cmd_data_download, "download raw files for sources that permit it"),
        ("parse", cmd_data_parse, "parse raw files into normalised verse records"),
        ("validate", cmd_data_validate, "validate parsed records"),
        ("build", cmd_data_build, "parse + validate + build the training corpus"),
    ):
        sub = data.add_parser(name, help=helptext)
        sub.add_argument("--config", help="data config (default configs/data.yaml)")
        if name in {"download", "parse", "validate"}:
            sub.add_argument(
                "--source", action="append", help="source id (repeatable); default: all eligible"
            )
        if name == "download":
            sub.add_argument("--force", action="store_true", help="re-download even if cached")
        if name == "validate":
            sub.add_argument("--no-reference", action="store_true", help="skip the reference verse counts")
        sub.set_defaults(func=func)
    ref = data.add_parser("reference", help="write reference verse counts from a metadata source")
    ref.add_argument("--config")
    ref.add_argument("--source", required=True)
    ref.set_defaults(func=cmd_data_reference)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_logging(args.log_level, json_console=args.log_json, log_file=args.log_file)
    except OSError as exc:
        print(f"error: cannot open log file {args.log_file}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        runtime = Runtime(args)
        code: int = args.func(runtime)
        return code
    except (ConfigError, ProjectRootNotFoundError) as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception as exc:  # report, don't dump a traceback unless debugging
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=args.log_level == "DEBUG")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
