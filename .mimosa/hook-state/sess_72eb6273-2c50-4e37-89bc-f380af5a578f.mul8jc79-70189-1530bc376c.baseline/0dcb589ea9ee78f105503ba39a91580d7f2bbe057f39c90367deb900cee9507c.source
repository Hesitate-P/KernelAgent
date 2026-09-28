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

"""Naive LLM-in-the-loop script for the Ascend C contest problem.

This is the low-risk fallback next to the KernelAgent beam-search route
(``examples/run_ascend_c_search.py``).  It keeps only the essentials:

    LLM proposes an edit to the source bundle
      -> CPU compile gate (free, no device)
      -> official build + correctness + public benchmark
      -> parse G_ref_public from artifacts/public_benchmark.json
      -> keep the best bundle, log the attempt, retry

Everything hard about measurement is shared with the skeleton route through
``triton_kernel_agent.platform.ascend_c`` (bundle format, repo slot, the
official harness driver, the NPU lock, the variant cache), so the two routes
stay comparable and can be mixed.

Examples
--------
CPU-only smoke test (no device, synthetic score)::

    ASCENDC_RSM_FAKE=1 ASCENDC_RSM_PREBUILD=1 \\
    python examples/naive_ascend_c_loop.py --template /path/to/repo \\
        --work-root /tmp/ka --iterations 2

Real run on a compute node::

    export SJTU_API_KEY=...
    ASCENDC_RSM_TEMPLATE=/scratch/rsm-template ASCENDC_RSM_WORK=/scratch/ka-work \\
    python examples/naive_ascend_c_loop.py \\
        --template "$ASCENDC_RSM_TEMPLATE" --work-root "$ASCENDC_RSM_WORK" \\
        --model deepseek-chat --iterations 8 --budget-seconds 1500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernel_agent.platform.ascend_c import (  # noqa: E402
    EDITABLE_FILES,
    AscendCConfig,
    VariantEvaluator,
    merge_bundle,
    parse_bundle,
    render_bundle,
    starting_bundle,
    starting_files,
    write_problem_statement,
)

# Deliberately coarse "lever" list: each iteration is told to attack exactly one
# of these, which keeps the proposals small enough to survive a single edit.
LEVERS = [
    "core occupancy: cases with fewer than 40 segments leave most AI Cores idle; "
    "split the segment's rows (or its D columns) across cores",
    "batched short segments: S=2048/4096 with mean length 8-32 pays per-segment "
    "DMA + reduction overhead; process several segments per vector operation",
    "DMA traffic: avoid re-reading score/x from GM; keep the max/sum passes in UB",
    "scalar serialisation: remove GetValue() calls from the per-row loop",
    "vector pipeline: replace pointer-chasing per-row Muls+Sub+Add with tiled "
    "vector ops, and drop redundant PipeBarrier<PIPE_ALL>",
    "tiling shape: pick kScoreTile/kRowsPerXTile to fit the 192 KB UB budget "
    "without spilling, and align D tiles to 32 B",
]

SYSTEM_PROMPT = """\
You are an expert Ascend C (CANN 8.5, Ascend 910B3) kernel engineer.
You edit a hand-written operator's sources so that it runs faster on the
official benchmark while still passing the official correctness gate.
You reply with complete files in a strict format and nothing else.
"""

EDIT_PROMPT = """\
## TASK
Improve the Ascend C implementation of "Ragged Softmax Moments" (per-segment
numerically stable softmax, weighted mean / rstd, logsumexp).

## SCORING
`G_ref_public` = weight-normalised geometric mean of `baseline_ms / your_ms`
over 7 public shapes. 1.0 = parity with the shipped baseline. A candidate counts
only if it compiles AND passes the official correctness suite (60 cases against
an FP64 oracle with tight tolerances: mean atol 3e-3, rstd rtol 4e-3,
logsumexp atol 5e-3).

## HARDER-CODED FACTS
- Only these files exist and may be edited (no new files):
{files}
- The kernel entry `rsm_solution(...)`, the `RsmSolutionTiling` layout, the
  `ConfigureSolutionLaunch(...)` signature and the `LaunchSolution(...)`
  argument order are called through macros from `runner/main.cpp`; keep them.
- `block_dim` is capped at 40 AI Cores; the workspace is <= 4 MiB; the inputs
  (score, x, offsets) are read-only; every output element must be written.
- S in [1, 8192], N in [S, 1048576], N*D <= 16777216, D in
  {{32,48,64,80,96,128,160,192,256}}, <= 16384 rows per segment,
  epsilon in [3e-6, 1e-2]. Do not assume only the public shapes are tested.

## MEASURED FEEDBACK
{report}

## CURRENT SOURCES
{bundle}

## THIS ITERATION'S FOCUS (do only this)
{focus}

## OUTPUT FORMAT (strict)
Emit the complete content of every file you changed:
<<<FILE solution/op_kernel/rsm_solution_core.h>>>
...full file...
<<<ENDFILE>>>
No prose, no markdown fences, no elisions. If you change a signature, update
every call site in every file you emit. Only name symbols that already exist in
the sources above or in `kernel_operator.h`.

Make the smallest change that plausibly moves the benchmark, and explain nothing.
"""


def load_env_file(path: Path | None) -> None:
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


def call_llm(model: str, prompt: str, max_tokens: int = 16384) -> str:
    """One LLM call through the shared retry/proxy-relaxation helper.

    Reusing ``ascend_c_components._llm_call`` instead of calling the provider
    directly means this fallback route inherits the same resilience (retries
    with backoff, dropping a proxy that cannot reach the gateway) as the
    beam-search route, so a flaky campus endpoint cannot silently zero out an
    iteration.
    """
    import logging

    from triton_kernel_agent.platform.ascend_c_components import _llm_call
    from utils.providers import get_model_provider

    provider = get_model_provider(model)
    return _llm_call(provider, model, [{"role": "user", "content": prompt}],
                     logging.getLogger("naive_ascend_c_loop"), max_tokens=max_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--budget-seconds", type=int, default=0,
                        help="stop cleanly after this many seconds")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--seed-dir", type=Path, default=None,
                        help="optional harness overlay applied to the pristine "
                             "sources (default: KernelAgent/examples/seeds/ascend_c_rsm)")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mind-change", action="store_true",
                        help="allow changes to the host/tiling files too "
                             "(default: core header only, the safest edit)")
    args = parser.parse_args()

    load_env_file(args.env_file)
    random.seed(args.seed)
    os.environ.setdefault("ASCENDC_RSM_TEMPLATE", str(args.template))
    os.environ.setdefault("ASCENDC_RSM_WORK", str(args.work_root))

    config = AscendCConfig.from_env(args.template, args.work_root)
    evaluator = VariantEvaluator(config)
    problem_path = args.work_root / "problem.md"
    write_problem_statement(args.template, problem_path)
    allowed = [EDITABLE_FILES[4]] if not args.mind_change else list(EDITABLE_FILES)
    allowed = [p for p in EDITABLE_FILES if p in allowed]

    log_path = args.work_root / "naive_loop.jsonl"
    state_path = args.work_root / "naive_state.json"

    # ---- starting point: best measured bundle, else the pristine template ----
    best_bundle = None
    best_g = 0.0
    for path in sorted((args.work_root / "variants").glob("*/result.json")) \
            if (args.work_root / "variants").is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("benchmark", {}).get("ok") and float(data.get("g_ref_public", 0)) > best_g:
            best_g = float(data["g_ref_public"])
            best_bundle = (path.parent / "bundle.txt").read_text(encoding="utf-8")
    if best_bundle is None:
        best_bundle = starting_bundle(args.template, args.seed_dir)
        print(f"starting from the pristine template ({len(best_bundle)} bytes)")
    else:
        print(f"resuming from a measured bundle: G_ref_public={best_g:.4f}x")

    deadline = (time.monotonic() + args.budget_seconds) if args.budget_seconds else None
    history: list[dict] = []
    started = time.monotonic()

    for iteration in range(1, args.iterations + 1):
        if deadline and time.monotonic() >= deadline:
            print(f"budget exhausted after {iteration - 1} iterations")
            break
        focus = LEVERS[(iteration - 1) % len(LEVERS)]
        prompt = EDIT_PROMPT.format(
            files="\n".join(f"  - {p}" for p in allowed),
            report=(evaluator.read_live_report() or "(no measurement yet)"),
            bundle=best_bundle,
            focus=focus,
        )
        (args.work_root / "naive_prompts").mkdir(parents=True, exist_ok=True)
        (args.work_root / "naive_prompts" / f"iter{iteration:03d}.txt").write_text(
            prompt, encoding="utf-8")

        print(f"\n=== iteration {iteration}: {focus[:60]}...")
        candidate = None
        try:
            reply = call_llm(args.model, prompt)
        except Exception as exc:
            print(f"  LLM call failed: {exc}")
            continue
        (args.work_root / "naive_prompts" / f"iter{iteration:03d}_reply.txt").write_text(
            reply, encoding="utf-8")
        files, problems = parse_bundle(reply)
        files = {p: body for p, body in files.items() if p in allowed}
        if not files:
            print(f"  reply carried no usable bundle: {problems[:2]}")
            history.append({"iteration": iteration, "focus": focus,
                            "outcome": "unparsed_reply"})
            continue
        candidate = merge_bundle(best_bundle, files)

        result = evaluator.evaluate(candidate)
        repairs = 0
        while not result.verified and repairs < args.repair_attempts:
            repairs += 1
            print(f"  gate failed ({result.error}); repair {repairs}/"
                  f"{args.repair_attempts}")
            repair_prompt = (
                "The candidate below failed the official Ascend C gate. Fix ONLY "
                "what is needed so it compiles and passes; keep the algorithm and "
                "all signatures intact, and update every call site you touch.\n\n"
                f"## FAILURE\n{result.build.output[-4000:] if not result.build.ok else result.correctness.output[-4000:]}\n\n"
                f"## CANDIDATE\n{candidate}\n\n"
                "Reply with the changed files in the <<<FILE path>>>...<<<ENDFILE>>> "
                "format, complete file bodies only."
            )
            try:
                reply = call_llm(args.model, repair_prompt)
            except Exception as exc:
                print(f"  repair LLM call failed: {exc}")
                break
            repair_files, _ = parse_bundle(reply)
            repair_files = {p: b for p, b in repair_files.items() if p in allowed}
            if not repair_files:
                break
            candidate = merge_bundle(candidate, repair_files)
            result = evaluator.evaluate(candidate)

        outcome = {
            "iteration": iteration,
            "focus": focus,
            "variant_id": result.variant_id,
            "verified": result.verified,
            "g_ref_public": result.g_ref_public,
            "objective_ms": result.objective_ms,
            "error": result.error,
            "repairs": repairs,
            "elapsed_s": round(time.monotonic() - started, 1),
        }
        history.append(outcome)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(outcome, sort_keys=True) + "\n")
        print(f"  -> variant {result.variant_id} verified={result.verified} "
              f"G={result.g_ref_public:.4f}x best={best_g:.4f}x")

        if result.verified and result.g_ref_public > best_g:
            best_g = result.g_ref_public
            best_bundle = candidate
            print(f"  ** new best: G_ref_public={best_g:.4f}x")
            (args.work_root / "naive_best_bundle.txt").write_text(
                best_bundle, encoding="utf-8")
            state_path.write_text(json.dumps({
                "variant_id": result.variant_id,
                "g_ref_public": best_g,
                "worst_case_speedup": result.worst_case,
                "per_case": result.per_case,
                "iterations": iteration,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"iterations={len(history)} best G_ref_public={best_g:.4f}x")
    print(f"best bundle: {args.work_root / 'naive_best_bundle.txt'}")
    print(f"attempt log: {log_path}")

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        files, _ = parse_bundle(best_bundle)
        for path, body in files.items():
            dest = args.export_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        (args.export_dir / "bundle.txt").write_text(best_bundle, encoding="utf-8")
        (args.export_dir / "summary.json").write_text(json.dumps({
            "g_ref_public": best_g, "history": history,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"exported to {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
