#!/usr/bin/env python3
"""Free-form LLM loop for the Ascend C contest problem (DeepSeek flash).

Per the 2026-09-25 operations directive the model gets FULL creative freedom:
whole-file rewrites, structural refactors, any optimisation it invents.  What
stays is the automatic verification chain (never a capability limit):

    CPU compile gate -> official correctness (49 public cases vs FP64 oracle)
    -> 24-case adversarial FP32-underflow battery -> runtime robustness
    -> quick public benchmark (G_ref_public) -> formal G + robustness for wins

and one correctness rule: the weight-sum underflow protection must keep its
BEHAVIOUR (a guard refactor is fine; deleting the protection without an
equivalent is rejected by the underflow battery and by a cheap name-count
pre-check that turns into repair feedback, not a hard stop).

The model may answer with complete files (``<<<FILE ...>>>``), SEARCH/REPLACE
patch blocks (``<<<PATCH ...>>>``), or a mix; both are applied verbatim.

Credentials: ``DEEPSEEK_API_KEY`` via the official platform API only
(api.deepseek.com, model ``deepseek-flash``); thinking is disabled because
with reasoning enabled the model burns its whole completion budget on
chain-of-thought and emits an EMPTY answer on this ~48 KB kernel prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernel_agent.platform.ascend_c import (  # noqa: E402
    AscendCConfig,
    VariantEvaluator,
    bundle_sanity,
    merge_bundle,
    parse_bundle,
    render_bundle,
    starting_bundle,
)

#: Name-count pre-check for the underflow protection.  A violation is fed
#: back to the model as repair input ("refactor is allowed, behaviour is
#: not negotiable"); the pz battery adjudicates the real semantics.
GUARD_MIN_COUNTS = {"kMinWeightSum": 12, "1e-30": 1}

PATCH_BLOCK_RE = re.compile(
    r"<<<PATCH\s+(.+?)>>>\s*\n(.*?)<<<ENDPATCH>>>", re.DOTALL)
HUNK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE\s*\n?",
    re.DOTALL)

#: Rotating directions offered to the model as NON-binding hints.  With
#: thinking disabled the model executes a direction well but free-form
#: prompts make it parrot the sources back unchanged (measured 4/6 identical
#: replies), so each iteration also offers one concrete angle to try.
DIRECTIONS = [
    "batch several short segments per core worker to amortise per-segment "
    "reduction overhead (many_short is 1.6x; S=2048, mean length 8)",
    "rebalance rows_per_core so all 40 cores work even when S < 40 "
    "(split one segment's rows across cores; medium is 2.7x)",
    "resize the UB tiles (score/x tile rows, vector widths) for the "
    "saturated shape (S=4096, mean length 32, currently 2.5x)",
    "move the single-kernel vs split-kernel threshold "
    "(kSingleKernelElements, today 49152) and re-tune block counts",
    "lower kMinRowsPerBlock (today 64) so small shapes use more cores",
    "unroll or software-pipeline the per-row weight accumulation loop",
    "double-buffer the score/x DMA transfers against compute",
    "improve the cross-core merge phase (fewer syncs, wider vectors)",
]

SYSTEM_PROMPT = """\
You are an expert Ascend C (CANN 8.5, Ascend 910B3) kernel engineer with
FULL creative freedom: whole-file rewrites, structural refactors, scheduling,
tiling, memory layout, instruction-level tuning - anything you judge moves
the score. An automatic verification chain tells you exactly what works.
"""

EDIT_PROMPT = """\
## TASK
Maximise `G_ref_public` for the "Ragged Softmax Moments" Ascend C operator.
Current best: {best_g:.4f}x. You are completely free in HOW: rewrite whole
files, restructure the kernel, change the algorithm's scheduling or memory
layout - your call. The per-case table shows where the headroom is (weakest:
many_short ~1.6x, saturated ~2.5x, medium ~2.7x; strongest already 20-50x),
but you decide what to attack and how.

## VERIFICATION CHAIN (automatic; it tells you right from wrong)
1. compile (CPU prebuild + official build)
2. 49 public correctness cases vs an FP64 oracle (mean atol 3e-3, rstd rtol
   4e-3, logsumexp atol 5e-3) + a 24-case adversarial FP32-underflow battery
   (score gaps ~85..110) + runtime robustness on mutated inputs
3. quick public benchmark -> G_ref_public
Anything that beats the best gets a formal re-measurement. When a step fails
you receive its full output (compiler errors, per-case numerical diffs) and
can iterate on it.

## THE ONE CORRECTNESS RULE (semantic, not stylistic)
Weight-sum underflow must never produce NaN/inf outputs. Today the core
header guarantees this via guards spelled `kMinWeightSum` (1e-30f). You may
refactor how the protection is written, but the behaviour must survive - the
underflow battery adjudicates automatically. Wholesale deletion without an
equivalent gets rejected and fed back to you.

## OTHER HARD FACTS
- Editable files (no new files): the files in the bundle below.
- `rsm_solution(...)`, `RsmSolutionTiling`, `ConfigureSolutionLaunch`,
  `LaunchSolution` are called through macros from runner/main.cpp; if you
  change their signatures, update every call site you emit.
- block_dim <= 40 AI cores; workspace <= 4 MiB; inputs read-only; every
  output element must be written.
- S in [1, 8192], N in [S, 1048576], N*D <= 16777216, D in
  {{32,48,64,80,96,128,160,192,256}}, <= 16384 rows per segment,
  epsilon in [3e-6, 1e-2]. The judge regenerates data: no shape special
  casing, no benchmark-value hardcoding.

## MEASURED PER-CASE TABLE (current best)
{report}

## PREVIOUS ATTEMPTS (automatic feedback; avoid repeating failures)
{history}

## SUGGESTED DIRECTION (a hint, not a constraint - your plan wins)
{direction}

## CURRENT SOURCES (multi-file bundle)
{bundle}

## OUTPUT FORMAT (strict, nothing else)
Complete rewritten files:
<<<FILE solution/op_kernel/rsm_solution_core.h>>>
...full file...
<<<ENDFILE>>>
and/or SEARCH/REPLACE patches (mixing is fine):
<<<PATCH solution/op_host/rsm_solution_host.cpp>>>
<<<<<<< SEARCH
...exact existing lines (verbatim, with indentation)...
=======
...replacement lines...
>>>>>>> REPLACE
<<<ENDPATCH>>>
Rules: patch SEARCH text must appear exactly once in its file; every emitted
file must be complete and self-contained C++; no prose outside blocks.
"""

REPAIR_PROMPT = """\
Your change was applied and the automatic verification chain rejected it.
You have full freedom again: fix the failure in whatever way you judge best
(small fix, different approach, or revert-and-redo). The FULL rejection
output is below - use it.

## REJECTION (full)
{failure}

## CURRENT SOURCES (after your previous change; SOURCE HAS MOVED)
{bundle}

## REMINDERS
- FILE blocks and/or PATCH blocks, same strict format, nothing else.
- Underflow protection behaviour must survive (refactor allowed).

Reply with blocks only.
"""


def load_env_file(path: Path | None) -> None:
    if path is None or not Path(path).is_file():
        return
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def call_llm(model: str, prompt: str, max_tokens: int = 8192) -> str:
    """One LLM call via the DeepSeek platform API (api.deepseek.com).

    Called directly - never through the model registry - because the
    registry resolves ``deepseek-*`` names to the SJTU campus gateway first,
    and that route is explicitly out of scope.  ``thinking: disabled`` is
    load-bearing (measured: with reasoning on, 8k/16k/32k reasoning tokens
    and an EMPTY answer on this task's ~48 KB kernel prompt).
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set: this loop only talks to the "
            "official DeepSeek platform API (api.deepseek.com) and refuses "
            "to fall back to any other endpoint")
    from openai import OpenAI

    from utils.providers.env_config import prune_unresolvable_proxies

    prune_unresolvable_proxies()
    client = OpenAI(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"], timeout=300, max_retries=4)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}})
            text = (response.choices[0].message.content or "").strip()
            if "</think>" in text:  # defensive: some routes inline reasoning
                text = text.split("</think>", 1)[1].strip()
            if text:
                return text
            last_error = RuntimeError("model returned an empty message")
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"deepseek call failed after 3 attempts: {last_error!r}")


def parse_patch(reply: str) -> dict[str, list[tuple[str, str]]]:
    """Parse the reply into {path: [(search, replace), ...]}."""
    patches: dict[str, list[tuple[str, str]]] = {}
    reply_norm = reply.replace("<<<ENDFILE-MARKER>>>", "<<<ENDPATCH>>>")
    for match in PATCH_BLOCK_RE.finditer(reply_norm):
        path = match.group(1).strip()
        hunks = [ (h.group(1), h.group(2))
                  for h in HUNK_RE.finditer(match.group(2) + "\n") ]
        if hunks:
            patches.setdefault(path, []).extend(hunks)
    return patches


def apply_patch(files: dict[str, str],
                patches: dict[str, list[tuple[str, str]]],
                ) -> tuple[dict[str, str], list[str]]:
    """Apply SEARCH/REPLACE hunks verbatim; return (new_files, problems)."""
    problems: list[str] = []
    out = dict(files)
    for path, hunks in patches.items():
        if path not in out:
            problems.append(f"{path}: not an editable file in the bundle")
            continue
        text = out[path]
        for i, (search, replace) in enumerate(hunks):
            occurrences = text.count(search)
            if occurrences == 0:
                problems.append(
                    f"{path} hunk {i + 1}: SEARCH text not found (copy it "
                    "verbatim, with indentation, from the current file)")
                continue
            if occurrences > 1:
                problems.append(
                    f"{path} hunk {i + 1}: SEARCH text matches {occurrences} "
                    "places; extend it with neighbouring lines until unique")
                continue
            text = text.replace(search, replace, 1)
        out[path] = text
    return out, problems


def guard_check(files: dict[str, str], parent_files: dict[str, str]) -> list[str]:
    """Underflow-guard name-count pre-check (repair feedback, not a hard stop)."""
    problems = []
    core = "solution/op_kernel/rsm_solution_core.h"
    new_text = files.get(core, "")
    old_text = parent_files.get(core, "")
    for marker, minimum in GUARD_MIN_COUNTS.items():
        have = new_text.count(marker)
        if have < minimum:
            problems.append(
                f"GUARD FEEDBACK: core.h now carries {have} occurrences of "
                f"`{marker}` (had {old_text.count(marker)}, minimum {minimum}). "
                "Refactoring the underflow protection is allowed, but the "
                "NaN-free behaviour is verified by the adversarial battery - "
                "either restore the guards or provide an equivalent "
                "protection, then resubmit")
    return problems


def failure_text(result) -> str:
    if result is None:
        return "the reply carried no parseable FILE or PATCH block"
    if not result.build.ok:
        return ("PHASE compile:\n" + (result.build.output or "")[-8000:])
    if not result.correctness.ok:
        return ("PHASE correctness/robustness/pz:\n"
                + (result.correctness.output or "")[-8000:])
    if not result.benchmark.ok:
        return ("PHASE benchmark:\n" + (result.benchmark.output or "")[-4000:])
    return str(result.error or "unknown failure")


def history_summary(work_root: Path, baseline_g: float) -> str:
    """One line per earlier attempt, so the model avoids reruns."""
    lines: list[str] = []
    path = work_root / "diff_loop.jsonl"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            target = data.get("target", "free")
            if data.get("verified"):
                g = float(data.get("g_ref_public", 0))
                verdict = ("FASTER, became the new best" if g > baseline_g
                           else "slower, reverted")
                lines.append(f"{target}: verified, G={g:.4f} ({verdict})")
            else:
                err = str(data.get("error", "reply/patch-format failure"))[:80]
                lines.append(f"{target}: rejected ({err})")
    return "\n".join(f"- {line}" for line in lines[-10:]) or "(none yet)"


def apply_reply(parent_bundle: str, reply: str
                ) -> tuple[dict[str, str] | None, list[str]]:
    """Apply a mixed FILE/PATCH reply onto the parent bundle's files."""
    parent_files, _ = parse_bundle(parent_bundle)
    emitted, problems = parse_bundle(reply)
    # parse_bundle complains about replies that carry no FILE block at all -
    # which is legitimate for a pure-PATCH reply, so drop that generic
    # complaint unless the reply has neither form.
    problems = [p for p in problems if "no <<<FILE" not in p]
    for path in emitted:
        if path not in parent_files:
            problems.append(f"{path}: not an editable file in the bundle")
    whole = {p: b for p, b in emitted.items() if p in parent_files}
    patches = parse_patch(reply)
    files = dict(parent_files)
    if whole:
        files.update(whole)
    if patches:
        files, patch_problems = apply_patch(files, patches)
        problems += patch_problems
    if not whole and not patches:
        return None, (problems or ["no <<<FILE>>> or <<<PATCH>>> block found"])
    return files, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--budget-seconds", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--writable", default="",
                        help="comma-separated editable paths (default: all 7)")
    args = parser.parse_args()

    load_env_file(args.env_file)
    os.environ.setdefault("ASCENDC_RSM_TEMPLATE", str(args.template))
    os.environ.setdefault("ASCENDC_RSM_WORK", str(args.work_root))

    config = AscendCConfig.from_env(args.template, args.work_root)
    evaluator = VariantEvaluator(config)
    if args.writable:
        config.writable_files = tuple(
            p.strip() for p in args.writable.split(",") if p.strip())

    log_path = args.work_root / "diff_loop.jsonl"
    (args.work_root / "diff_prompts").mkdir(parents=True, exist_ok=True)

    # ---- start from the best measured bundle (px-f1 baseline expected) ----
    best_bundle, best_g = None, 0.0
    variants_dir = args.work_root / "variants"
    if variants_dir.is_dir():
        for path in sorted(variants_dir.glob("*/result.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (data.get("benchmark", {}).get("ok")
                    and float(data.get("g_ref_public", 0)) > best_g):
                best_g = float(data["g_ref_public"])
                best_bundle = (path.parent / "bundle.txt").read_text(encoding="utf-8")
    if best_bundle is None:
        best_bundle = starting_bundle(args.template)
        print("starting from the pristine template")
    else:
        print(f"resuming from measured bundle: G_ref_public={best_g:.4f}x")

    deadline = (time.monotonic() + args.budget_seconds) if args.budget_seconds else None
    history: list[dict] = []
    started = time.monotonic()

    for iteration in range(1, args.iterations + 1):
        if deadline and time.monotonic() >= deadline:
            print(f"budget exhausted after {iteration - 1} iterations")
            break
        report = evaluator.read_live_report() or "(no measurement yet)"
        direction = DIRECTIONS[(iteration - 1) % len(DIRECTIONS)]
        prompt = EDIT_PROMPT.format(
            best_g=best_g, report=report, bundle=best_bundle,
            history=history_summary(args.work_root, best_g),
            direction=direction)
        (args.work_root / "diff_prompts" / f"iter{iteration:03d}.txt").write_text(
            prompt, encoding="utf-8")

        print(f"\n=== iteration {iteration} [{args.model}] (free-form)")
        result = None
        candidate = None
        attempt = 0
        base_bundle = best_bundle
        while attempt <= args.repair_attempts:
            try:
                reply = call_llm(args.model, prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"  LLM call failed: {exc}")
                history.append({"iteration": iteration, "target": "free",
                                "outcome": "llm_error", "error": str(exc)[:200]})
                break
            (args.work_root / "diff_prompts" /
             f"iter{iteration:03d}_reply{attempt}.txt").write_text(
                reply, encoding="utf-8")
            files, problems = apply_reply(base_bundle, reply)
            if files is None:
                print(f"  unparsed reply: {problems[0][:120]}")
                if attempt >= args.repair_attempts:
                    history.append({"iteration": iteration, "target": "free",
                                    "outcome": "unparsed_reply"})
                    break
                prompt = REPAIR_PROMPT.format(
                    failure="\n".join(problems), bundle=base_bundle)
                attempt += 1
                continue

            guard_notes = guard_check(files, parse_bundle(base_bundle)[0])
            if guard_notes:
                # Soft rejection: the semantic adjudication is the battery's
                # job; the count check just guarantees we send the model a
                # candidate that can plausibly pass it.
                print(f"  guard feedback: {guard_notes[0][:110]}")
                if attempt >= args.repair_attempts:
                    history.append({"iteration": iteration, "target": "free",
                                    "outcome": "guard_feedback_unrepaired"})
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "iteration": iteration, "target": "free",
                            "outcome": "guard_feedback_unrepaired",
                            "elapsed_s": round(time.monotonic() - started, 1),
                        }, sort_keys=True) + "\n")
                    break
                prompt = REPAIR_PROMPT.format(
                    failure="\n".join(guard_notes), bundle=render_bundle(files))
                attempt += 1
                continue

            structural = bundle_sanity(base_bundle, files)
            if structural:
                print(f"  structural: {structural[0][:110]}")
                if attempt >= args.repair_attempts:
                    break
                prompt = REPAIR_PROMPT.format(
                    failure="\n".join(structural), bundle=render_bundle(files))
                attempt += 1
                continue

            candidate = render_bundle(files)
            result = evaluator.evaluate(candidate)
            if result.verified:
                break
            print(f"  gate failed: {str(result.error)[:140]}")
            if attempt >= args.repair_attempts:
                break
            prompt = REPAIR_PROMPT.format(
                failure=failure_text(result), bundle=candidate)
            attempt += 1

        if result is not None and result.verified:
            outcome = {
                "iteration": iteration, "target": "free",
                "variant_id": result.variant_id, "verified": True,
                "g_ref_public": result.g_ref_public,
                "worst_case": result.worst_case,
                "elapsed_s": round(time.monotonic() - started, 1),
            }
            history.append(outcome)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(outcome, sort_keys=True) + "\n")
            print(f"  -> {result.variant_id} verified G={result.g_ref_public:.4f}x "
                  f"(best {best_g:.4f}x)")
            if result.g_ref_public > best_g:
                best_g = result.g_ref_public
                best_bundle = candidate
                (args.work_root / "diff_best_bundle.txt").write_text(
                    best_bundle, encoding="utf-8")
                (args.work_root / "diff_state.json").write_text(json.dumps({
                    "variant_id": result.variant_id,
                    "g_ref_public": best_g,
                    "worst_case_speedup": result.worst_case,
                    "per_case": result.per_case,
                    "history": history,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"  ** new best: G_ref_public={best_g:.4f}x")
        elif result is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "iteration": iteration, "target": "free",
                    "outcome": "gate_failed",
                    "error": str(result.error)[:300],
                    "elapsed_s": round(time.monotonic() - started, 1),
                }, sort_keys=True) + "\n")

    print("\n" + "=" * 70)
    print(f"iterations={len(history)} best G_ref_public={best_g:.4f}x")
    print(f"attempt log: {log_path}")
    if args.export_dir and best_bundle:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for path, body in parse_bundle(best_bundle)[0].items():
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
