#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drive the KernelAgent skeleton over the Ascend C contest problem.

Pipeline per round (beam search lives in ``OptimizationManager``):

    candidate parents -> N workers -> [LLM rewrites a source bundle]
                                   -> [official build + correctness gate]
                                   -> [official public benchmark -> G_ref_public]
                                   -> beam update + reflexion

Everything the workers need (where the pristine repo is, where variants are
materialised, how strict the gate is) is passed through the ``ASCENDC_RSM_*``
environment variables so it survives the ``multiprocessing`` spawn cleanly.

Examples
--------
Dry run (no LLM, no device)::

    python examples/run_ascend_c_search.py --template /path/to/repo \\
        --work-root /tmp/ka --dry-run

Full search::

    export SJTU_API_KEY=...
    export ASCENDC_RSM_TEMPLATE=/scratch/rsm-template
    export ASCENDC_RSM_WORK=/scratch/ka-work
    python examples/run_ascend_c_search.py \\
        --template "$ASCENDC_RSM_TEMPLATE" --work-root "$ASCENDC_RSM_WORK" \\
        --config examples/configs/ascend_c_rsm.yaml --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernel_agent.opt_manager import OptimizationManager  # noqa: E402
from triton_kernel_agent.platform.ascend_c import (  # noqa: E402
    EDITABLE_FILES,
    AscendCConfig,
    VariantEvaluator,
    parse_bundle,
    render_bundle,
    seed_overlay,
    starting_bundle,
    write_problem_statement,
)


def load_env_file(path: Path | None) -> None:
    """Populate ``os.environ`` from a dotenv file without overriding it."""
    if path is None or not Path(path).is_file():
        return
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def initial_bundle(template_dir: Path, seed_dir: Path | None = None) -> str:
    """Pristine editable files, with the optional harness overlay applied."""
    return starting_bundle(template_dir, seed_dir)


def resolve_config(config_path: Path) -> dict:
    """Load the YAML config and absolutise template override paths."""
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    templates = dict(data.get("templates") or {})
    for key, value in templates.items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        templates[key] = str(candidate)
    if templates:
        data["templates"] = templates
    return data


def ranked_variants(evaluator: VariantEvaluator, limit: int) -> list[dict]:
    """Measured variants ordered best-first (source of the resume state)."""
    rows = []
    if not evaluator.variants_dir.is_dir():
        return rows
    for path in sorted(evaluator.variants_dir.glob("*/result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data.get("benchmark", {}).get("ok"):
            continue
        bundle = path.parent / "bundle.txt"
        if not bundle.is_file():
            continue
        rows.append({
            "variant_id": data.get("variant_id"),
            "objective_ms": float(data.get("objective_ms", float("inf"))),
            "g_ref_public": float(data.get("g_ref_public", 0.0)),
            "bundle": bundle.read_text(encoding="utf-8"),
        })
    rows.sort(key=lambda r: r["objective_ms"])
    return rows[:limit]


def load_reflexions(log_dir: Path) -> list[dict]:
    path = log_dir / "shared_reflexions.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_reflexions(log_dir: Path, manager) -> None:
    try:
        (log_dir / "shared_reflexions.json").write_text(
            json.dumps(manager.shared_reflexions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def export_best(work_root: Path, destination: Path, evaluator: VariantEvaluator) -> dict:
    """Copy the best measured variant's sources plus a patch to *destination*."""
    best = evaluator.best_so_far()
    if best is None:
        return {"exported": False, "reason": "no variant completed the benchmark"}
    variant_id = best["variant_id"]
    source = evaluator.variant_dir(variant_id)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("bundle.txt", "result.json", "public_benchmark.json",
                 "public_correctness_results.json"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
    if (source / "files").is_dir():
        shutil.copytree(source / "files", destination / "files")

    # Unified diff against the pristine template's editable files: this is the
    # artefact that can be replayed onto a fresh checkout of the problem.
    import difflib

    def _lines(text: str) -> list[str]:
        # The bundle always terminates files with "\n" while some shipped files
        # do not, so normalise before diffing - otherwise an unchanged file
        # looks changed and difflib can glue the last line of one file onto the
        # header of the next, producing a patch that does not apply.
        normalised = text.replace("\r\n", "\n").rstrip("\n")
        return [line + "\n" for line in normalised.split("\n")]

    patch_lines: list[str] = []
    changed: list[str] = []
    for relative in EDITABLE_FILES:
        pristine = (evaluator.config.template_dir / relative).read_text(encoding="utf-8")
        new_path = source / "files" / relative
        if not new_path.is_file():
            continue
        updated = new_path.read_text(encoding="utf-8")
        if (_lines(pristine) == _lines(updated)):
            continue
        changed.append(relative)
        patch_lines.extend(difflib.unified_diff(
            _lines(pristine), _lines(updated),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
    patch_text = "".join(patch_lines)
    (destination / "rsm-best.patch").write_text(patch_text, encoding="utf-8")

    summary = {
        "variant_id": variant_id,
        "g_ref_public": best.get("g_ref_public"),
        "worst_case_speedup": best.get("worst_case"),
        "objective_ms": best.get("objective_ms"),
        "build_seconds": best.get("build", {}).get("seconds"),
        "correctness_seconds": best.get("correctness", {}).get("seconds"),
        "benchmark_seconds": best.get("benchmark", {}).get("seconds"),
        "per_case": best.get("per_case"),
        "changed_files": changed,
        "patch_bytes": len(patch_text),
        "exported": True,
        "destination": str(destination),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True,
                        help="pristine contest repository copy")
    parser.add_argument("--work-root", type=Path, required=True,
                        help="where slots/variants/state/logs live")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "examples/configs/ascend_c_rsm.yaml")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--seed-dir", type=Path, default=None,
                        help="optional harness overlay applied to the pristine "
                             "sources before the search starts "
                             "(default: KernelAgent/examples/seeds/ascend_c_rsm)")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="dotenv file providing SJTU_API_KEY")
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="copy the best variant here when the search ends")
    parser.add_argument("--export-only", action="store_true",
                        help="skip the search; just refresh the export from the "
                             "measurements already recorded in the work root")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the setup (bundle/build/gate) and exit")
    parser.add_argument("--skip-initial-eval", action="store_true",
                        help="do not measure the unmodified template first")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore previously measured variants and reflexions")
    parser.add_argument("--resume-beam", type=int, default=2,
                        help="how many of the best measured bundles seed the beam")
    parser.add_argument("--max-seconds", type=int,
                        default=int(os.environ.get("ASCENDC_RSM_MAX_SECONDS", "0") or 0),
                        help="stop starting new rounds after this many seconds "
                             "(keeps the NPU job short; the cache makes the next "
                             "run resume where this one stopped)")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not os.environ.get("SJTU_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: no SJTU_API_KEY/DEEPSEEK_API_KEY in the environment; "
              "every LLM call will fail", file=sys.stderr)

    args.work_root.mkdir(parents=True, exist_ok=True)
    for key, value in (
        ("ASCENDC_RSM_TEMPLATE", args.template),
        ("ASCENDC_RSM_WORK", args.work_root),
    ):
        os.environ[key] = str(value)
        print(f"{key}={value}")

    config = AscendCConfig.from_env(args.template, args.work_root)
    evaluator = VariantEvaluator(config)
    problem_path = args.work_root / "problem.md"
    write_problem_statement(args.template, problem_path)
    overlay = seed_overlay(args.template, args.seed_dir)
    if overlay:
        print(f"seed harness overlay: {', '.join(sorted(overlay))}")
    else:
        from triton_kernel_agent.platform.ascend_c import harness_present

        reason = ("template already carries the harness"
                  if harness_present(args.template) else "no overlay found")
        print(f"seed harness overlay: not applied ({reason})")
    bundle = initial_bundle(args.template, args.seed_dir)
    print(f"initial bundle: {len(bundle)} bytes over {len(EDITABLE_FILES)} files")
    print(f"problem statement: {problem_path}")
    print(f"correctness gate: {config.correctness_mode} "
          f"({len(evaluator.slot.correctness_case_ids())} cases), "
          f"benchmark: {config.benchmark_mode}")

    if args.dry_run:
        parsed, problems = __import__(
            "triton_kernel_agent.platform.ascend_c", fromlist=["parse_bundle"]
        ).parse_bundle(bundle)
        print(f"round-trip parse: {len(parsed)} files, problems={problems}")
        result = evaluator.evaluate(bundle)
        print(json.dumps(result.to_dict(), indent=2)[:4000])
        return 0 if result.verified else 1

    if args.export_only:
        summary = export_best(args.work_root, args.export_dir or (args.work_root / "best"),
                              evaluator)
        print(json.dumps(summary, indent=2, sort_keys=True)[:3000])
        return 0 if summary.get("exported") else 1

    log_dir = args.log_dir or (args.work_root / "search")
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume state: best measured bundles + reflexions from earlier jobs ----
    seeds: list[dict] = []
    if not args.no_resume:
        seeds = ranked_variants(evaluator, max(1, args.resume_beam))
    if seeds:
        print(f"resuming: {len(seeds)} measured candidate(s), best "
              f"G_ref_public={seeds[0]['g_ref_public']:.4f}x "
              f"({seeds[0]['variant_id']})")
        bundle = seeds[0]["bundle"]
    prior_reflexions = [] if args.no_resume else load_reflexions(log_dir)
    if prior_reflexions:
        print(f"resuming: {len(prior_reflexions)} prior reflexion(s)")

    manager_config = resolve_config(args.config)
    manager_config["log_dir"] = str(log_dir)
    manager_config["database_path"] = str(log_dir / "program_database.json")
    manager_config["max_rounds"] = args.max_rounds

    if args.skip_initial_eval:
        manager_config["verifier"] = "noop"

    manager = OptimizationManager(**manager_config)
    print(f"search log dir: {log_dir}")
    print(f"platform: {manager_config.get('platform')}")

    # Seed the beam with the previously best bundles so consecutive short NPU
    # jobs continue an actual beam instead of restarting from the template.
    if len(seeds) > 1:
        from triton_kernel_agent.opt_worker_component.searching.history.models import (
            ProgramEntry,
            ProgramMetrics,
        )

        strategy = manager.strategy
        base_initialize = strategy.initialize

        def initialize_with_seeds(initial_program):
            base_initialize(initial_program)
            restored = [
                ProgramEntry(
                    program_id=f"resume_{i}_{seed['variant_id']}",
                    kernel_code=seed["bundle"],
                    metrics=ProgramMetrics(time_ms=seed["objective_ms"]),
                    problem_id=initial_program.problem_id,
                    generation=0,
                )
                for i, seed in enumerate(seeds)
            ]
            merged = restored + [
                entry for entry in strategy.top_kernels
                if entry.kernel_code not in {s["bundle"] for s in seeds}
            ]
            strategy.top_kernels = merged[: strategy.num_top_kernels]
            manager.logger.info(
                f"Beam seeded with {len(strategy.top_kernels)} resumed member(s)"
            )

        strategy.initialize = initialize_with_seeds

    deadline = (time.monotonic() + args.max_seconds) if args.max_seconds > 0 else None
    if deadline:
        print(f"time budget: {args.max_seconds}s (new rounds stop after that)")
        original_run_workers = manager._run_workers
        original_should_terminate = manager.strategy.should_terminate
        exhausted = {"flag": False}

        def budgeted_run_workers(candidates, round_num, problem_file, test_code,
                                 pytorch_baseline):
            if time.monotonic() >= deadline:
                exhausted["flag"] = True
                manager.logger.warning(
                    f"time budget exhausted before round {round_num}; stopping "
                    "(every completed measurement is cached, so the next run "
                    "resumes from the recorded best)"
                )
                return []
            return original_run_workers(candidates, round_num, problem_file,
                                        test_code, pytorch_baseline)

        def budgeted_should_terminate(round_num, max_rounds):
            if exhausted["flag"]:
                return True
            return original_should_terminate(round_num, max_rounds)

        manager._run_workers = budgeted_run_workers
        manager.strategy.should_terminate = budgeted_should_terminate

    manager.shared_reflexions.extend(prior_reflexions)
    result = manager.run_optimization(
        initial_kernel=bundle,
        problem_file=problem_path,
        test_code=[problem_path.read_text(encoding="utf-8")],
        max_rounds=args.max_rounds,
    )
    save_reflexions(log_dir, manager)

    print("=" * 78)
    print(f"success={result.get('success')} rounds={result.get('total_rounds')}")
    print(f"initial objective={result.get('initial_kernel_time_ms')} ms")
    print(f"best objective={result.get('best_time_ms')} ms")
    if result.get("initial_kernel_time_ms") and result.get("best_time_ms"):
        initial, best = result["initial_kernel_time_ms"], result["best_time_ms"]
        if best > 0 and initial > 0:
            print(f"objective improvement: {initial / best:.3f}x "
                  f"(equivalently G_ref_public {initial / best:.3f}x higher than the "
                  f"initial candidate's)")
    best = evaluator.best_so_far()
    if best:
        print(f"best variant: {best['variant_id']} G_ref_public="
              f"{best['g_ref_public']:.4f}x worst={best['worst_case']:.4f}x")

    if args.export_dir:
        summary = export_best(args.work_root, args.export_dir, evaluator)
        print(f"export: {json.dumps(summary)[:1500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
