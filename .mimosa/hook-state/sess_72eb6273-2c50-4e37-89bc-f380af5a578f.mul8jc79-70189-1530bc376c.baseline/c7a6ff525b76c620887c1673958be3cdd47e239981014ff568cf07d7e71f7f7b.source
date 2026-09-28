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

"""Ascend C ("ascend_c") platform components for the KernelAgent skeleton.

These are the *injectable* halves of the plugin.  They are registered in
:mod:`triton_kernel_agent.platform.registry` under the name ``"ascend_c"`` and
selected from the YAML ``platform:`` block, exactly like the built-in
``nvidia``/``ascend``/``noop`` backends::

    platform:
      verifier: ascend_c          # manager: initial-candidate gate
      benchmarker: ascend_c       # manager: initial-candidate measurement
      worker_runner: ascend_c     # manager: spawns Ascend C workers
      worker_benchmarker: ascend_c      # worker: orchestrator-facing Benchmark
      verification_worker: ascend_c     # worker: orchestrator-facing verifier
      specs_provider: ascend_c
      profiler: ascend_c          # no NCU on Ascend -> timing-only
      roofline_analyzer: ascend_c
      bottleneck_analyzer: ascend_c     # LLM source-level analysis
      rag_prescriber: ascend_c    # live-measurement report channel

What each seam replaces in the Triton design:

======================  ==================================  ==============================
skeleton seam           Triton behaviour                    Ascend C behaviour
======================  ==================================  ==============================
kernel unit             one ``.py`` string                  a multi-file source bundle
verifier                run ``test_kernel.py`` on GPU       build + official correctness
benchmarker             ``do_bench`` CUDA/HIP events        official public benchmark (G)
profiler                NCU counters                        none (timing-only)
bottleneck analysis     NCU metrics -> LLM                  source + measured case table
======================  ==================================  ==============================
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.ascend_c import (
    EDITABLE_FILES,
    AscendCConfig,
    EvalResult,
    StepResult,
    VariantEvaluator,
    bundle_id,
    bundle_sanity,
    parse_bundle,
    render_bundle,
)
from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RAGPrescriberBase,
    RooflineAnalyzerBase,
    RooflineResult,
    WorkerRunner,
)

_ASCEND_910B3_SPECS: dict[str, Any] = {
    "name": "Ascend 910B3 (Atlas 800T A2)",
    "architecture": "Ascend Da Vinci (Cube + Vector)",
    "peak_fp16_tflops": 313.0,
    "peak_bf16_tflops": 313.0,
    "peak_fp32_tflops": 78.0,
    "peak_memory_bw_gbps": 1600,
    "sm_count": 40,
    "ai_core_count": 40,
    "vector_core_count": 40,
    "max_threads_per_sm": 1024,
    "l1_cache_kb": 192,
    "l2_cache_mb": 32,
    "memory_gb": 64,
    "memory_type": "HBM2e",
    "ub_size_kb": 256,
    "notes": (
        "40 AI Cores per die are exposed to Ascend C as block_dim; each core has a "
        "192KB-class L1/UB tile buffer. Ascend C is an explicit SPMD/CUDA-like "
        "model: per-core local memory (TBuf/TQue), explicit GM<->UB DMA "
        "(DataCopy/DataCopyPad), vector intrinsics (Cast/Mul/Exp/Ln/Rsqrt/"
        "ReduceMax/ReduceSum), and inter-core communication via SyncAll / "
        "CrossCoreSetFlag / CrossCoreWaitFlag or atomics (SetAtomicAdd)."
    ),
}

#: Correctness cases whose history is most informative when a gate fails.
_GATE_HINT = (
    "public_precision_", "public_adversarial_", "public_regression_",
    "public_low_weight_outlier", "public_large_sparse", "public_underflow",
    "public_cross_grouped", "public_stable_softmax", "public_epsilon_",
)


# ---------------------------------------------------------------------------
# Shared evaluator construction
# ---------------------------------------------------------------------------


def _make_evaluator(kwargs: dict[str, Any] | None = None,
                    logger: logging.Logger | None = None,
                    slot_name: str = "manager") -> VariantEvaluator:
    kwargs = kwargs or {}
    config = AscendCConfig.from_env(
        template_dir=kwargs.get("template_dir") or kwargs.get("ascend_c_template"),
        work_root=kwargs.get("work_root") or kwargs.get("ascend_c_work"),
    )
    return VariantEvaluator(config, lock=None, logger=logger, slot_name=slot_name)


def _bundle_from_arg(value: Any) -> str:
    """Accept a bundle string or a path to a file containing one."""
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    text = str(value)
    if "\n" not in text and Path(text).is_file():
        return Path(text).read_text(encoding="utf-8")
    return text


def _llm_call(provider: Any, model: str, messages: list[dict[str, str]],
              logger: logging.Logger, max_tokens: int = 16384,
              attempts: int | None = None, base_delay: float | None = None) -> str:
    """Call the LLM with retries.

    The campus gateway is shared with the other contest tasks and does drop
    connections under load (observed: ``APIConnectionError('Connection error.')``
    on every worker at once while the LLM-serving job was running; an un-retried
    call loses a whole optimisation round, because a worker whose *analysis*
    call fails silently does no work at all).  Retries are cheap next to a lost
    round, so the default is generous and tunable:

    * ``ASCENDC_RSM_LLM_ATTEMPTS`` (default 6)
    * ``ASCENDC_RSM_LLM_BACKOFF``  (default 5 seconds, scaled by the attempt)
    """
    attempts = (int(os.environ.get("ASCENDC_RSM_LLM_ATTEMPTS", "6"))
                if attempts is None else attempts)
    base_delay = (float(os.environ.get("ASCENDC_RSM_LLM_BACKOFF", "5"))
                  if base_delay is None else base_delay)
    last_error: Exception | None = None
    proxy_relaxed = False
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = provider.get_response(model, messages, max_tokens=max_tokens)
            return response.content
        except Exception as exc:  # network, rate limit, truncated stream, ...
            last_error = exc
            if not proxy_relaxed and "Connection" in type(exc).__name__:
                # A proxy may resolve yet still not forward the campus gateway.
                # Drop it once so the retries can take the direct route instead.
                proxy_relaxed = bool(
                    getattr(provider, "drop_proxy_and_rebuild", lambda: False)()
                )
            if attempt >= attempts:
                break
            delay = base_delay * attempt
            logger.warning(f"LLM call failed ({exc!r}); retry {attempt}/{attempts - 1} "
                           f"in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error!r}")


# ---------------------------------------------------------------------------
# Manager-level: verifier
# ---------------------------------------------------------------------------


class AscendCVerifier(KernelVerifier):
    """Gates a candidate through the official build + correctness harness."""

    def __init__(self, log_dir: Path | str | None = None,
                 logger: logging.Logger | None = None,
                 ascend_c: dict[str, Any] | None = None, **_ignored: Any) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.evaluator = _make_evaluator(ascend_c, self.logger)

    def verify(self, kernel_code: str, problem_file: Path,
               test_code: list[str]) -> bool:
        result = self.evaluator.evaluate(_bundle_from_arg(kernel_code))
        if not result.verified:
            self.logger.error(
                "Initial Ascend C candidate failed the official gate: "
                f"{result.error}\n{result.build.output[-1500:] if not result.build.ok else result.correctness.output[-1500:]}"
            )
            return False
        self.logger.info(
            f"Initial Ascend C candidate passed the official gate "
            f"(G_ref_public={result.g_ref_public:.4f}x)"
        )
        return True


# ---------------------------------------------------------------------------
# Manager-level: benchmarker
# ---------------------------------------------------------------------------


class AscendCBenchmarker(KernelBenchmarker):
    """Measures a bundle with the official public benchmark.

    ``benchmark_kernel`` returns the *weighted geometric mean of the per-case
    solution latencies in milliseconds*.  That is a real duration, it is
    strictly monotone in ``1 / G_ref_public``, and it is the quantity the beam
    search minimises.  There is no eager PyTorch reference for an Ascend C
    operator, so the reference hooks report ``inf`` (the skeleton tolerates
    that everywhere).
    """

    def __init__(self, log_dir: Path | str | None = None,
                 logger: logging.Logger | None = None,
                 ascend_c: dict[str, Any] | None = None, **_ignored: Any) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.evaluator = _make_evaluator(ascend_c, self.logger)

    def evaluate_bundle(self, bundle: str) -> EvalResult:
        return self.evaluator.evaluate(_bundle_from_arg(bundle))

    def benchmark_kernel(self, kernel_code: str, problem_file: Path) -> float:
        result = self.evaluate_bundle(kernel_code)
        if not result.success:
            self.logger.warning(
                f"Ascend C candidate {result.variant_id} not measurable "
                f"({result.error or 'unknown error'}); reporting inf"
            )
            return float("inf")
        self.logger.info(
            f"Ascend C candidate {result.variant_id}: G_ref_public="
            f"{result.g_ref_public:.4f}x objective={result.objective_ms:.4f}ms"
        )
        return result.objective_ms

    def benchmark_reference(self, problem_file: Path) -> float:
        return float("inf")

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        return float("inf")


# ---------------------------------------------------------------------------
# Worker-level: orchestrator-facing benchmark ("Benchmark" duck type)
# ---------------------------------------------------------------------------


class AscendCWorkerBenchmark:
    """``Benchmark``-compatible facade used by ``OptimizationOrchestrator``.

    The orchestrator only needs ``benchmark_kernel(file, problem) -> {"time_ms"}``
    and ``benchmark_pytorch``; both are satisfied here, so the device-specific
    ``Benchmark`` class (Triton/``torch.npu``) never has to be imported.
    """

    def __init__(self, logger: logging.Logger | None = None,
                 ascend_c: dict[str, Any] | None = None,
                 log_dir: Path | str | None = None,
                 artifacts_dir: Path | str | None = None,
                 worker_id: int = 0,
                 **_ignored: Any) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.worker_id = worker_id
        self.evaluator = _make_evaluator(ascend_c, self.logger,
                                         slot_name=f"w{worker_id}")
        self._last: EvalResult | None = None

    @property
    def last_result(self) -> EvalResult | None:
        return self._last

    def benchmark_kernel(self, kernel_file: Path | str, problem_file: Path) -> dict[str, Any]:
        bundle = _bundle_from_arg(kernel_file)
        result = self.evaluator.evaluate(bundle)
        self._last = result
        if not result.success:
            self.logger.warning(
                f"[ascend_c] candidate {result.variant_id} not measurable: "
                f"{result.error or 'unknown'}"
            )
            return {
                "time_ms": float("inf"),
                "g_ref_public": result.g_ref_public,
                "error": result.error,
                "verdict": result.error or "evaluation_failed",
                "ptx_hash": None,
            }
        return {
            "time_ms": result.objective_ms,
            "g_ref_public": result.g_ref_public,
            "worst_case_speedup": result.worst_case,
            "ptx_hash": result.variant_id,
            "verdict": "measured",
        }

    def benchmark_pytorch(self, problem_file: Path) -> dict[str, float]:
        return {"time_ms": float("inf")}

    def benchmark_pytorch_compile(self, problem_file: Path) -> dict[str, float]:
        return {"time_ms": float("inf")}


# ---------------------------------------------------------------------------
# Worker-level: orchestrator-facing verifier ("VerificationWorker" duck type)
# ---------------------------------------------------------------------------


class AscendCVerificationWorker:
    """``VerificationWorker``-compatible gate for multi-file Ascend C source.

    Verification *is* the official harness: build the candidate, then run the
    public correctness + robustness suite.  On failure the compiler/verifier
    log is fed back to the LLM for a bounded number of repair attempts, which
    is the Ascend C analogue of Triton's ``verify_with_refinement`` loop.
    """

    def __init__(self, worker_id: int = 0, workdir: Path | str | None = None,
                 log_dir: Path | str | None = None,
                 openai_model: str = "deepseek-chat",
                 high_reasoning_effort: bool = False,
                 target_platform: str = "npu",
                 max_refine_attempts: int | None = None,
                 ascend_c: dict[str, Any] | None = None,
                 **_ignored: Any) -> None:
        self.worker_id = worker_id
        self.workdir = Path(workdir) if workdir else Path(".")
        self.log_dir = Path(log_dir) if log_dir else self.workdir
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.max_refine_attempts = int(
            max_refine_attempts if max_refine_attempts is not None
            else os.environ.get("ASCENDC_RSM_REFINE_ATTEMPTS", "1")
        )
        self.logger = logging.getLogger(f"ascend_c_verifier_{worker_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(self.log_dir / f"ascendc_verify_{worker_id}.log")
                handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )
                self.logger.addHandler(handler)
            except OSError:
                pass
        self.evaluator = _make_evaluator(ascend_c, self.logger,
                                         slot_name=f"w{worker_id}")
        self.provider: Any | None = None
        try:
            from utils.providers import get_model_provider

            self.provider = get_model_provider(self.openai_model)
        except Exception as exc:  # pragma: no cover - offline environments
            self.logger.warning(f"LLM provider unavailable: {exc}")

    # -- LLM plumbing (mirrors VerificationWorker) --------------------------

    def _call_llm(self, messages: list, **kwargs: Any) -> str:
        if self.provider is None:
            raise RuntimeError(f"No provider available for model {self.openai_model}")
        max_tokens = int(kwargs.get("max_tokens", 16384))
        return _llm_call(self.provider, self.openai_model, messages, self.logger,
                         max_tokens=max_tokens)

    def _extract_code_from_response(self, response_text: str, **_ignored: Any) -> str | None:
        """Return the canonical bundle carried by an LLM reply (or None)."""
        if not response_text:
            return None
        files, problems = parse_bundle(response_text)
        if not files:
            self.logger.warning(f"No bundle block found in the reply: {problems[:2]}")
            return None
        return render_bundle(files)

    # -- the gate -----------------------------------------------------------

    def _validate(self, bundle_text: str) -> tuple[dict[str, str], list[str]]:
        """Parse a bundle and reject writes outside the allowed file set."""
        files, problems = parse_bundle(bundle_text)
        if not problems:
            forbidden = sorted(set(files) - set(self.evaluator.config.writable_files))
            if forbidden:
                problems = [
                    f"{path!r} may not be emitted this run (writable: "
                    + ", ".join(self.evaluator.config.writable_files) + ")"
                    for path in forbidden
                ]
        return files, problems

    def verify_with_refinement(self, kernel_code: str, test_code: list[str],
                               problem_description: str,
                               max_refine_attempts: int | None = None,
                               parent_kernel: str | None = None
                               ) -> tuple[bool, str, str]:
        """Gate an LLM edit of the source bundle.

        ``kernel_code`` is the model's *reply*, which by design only contains the
        files it changed; ``parent_kernel`` is the candidate those changes apply
        to.  Everything downstream (structural check, compile gate, official
        harness) works on the merged bundle - evaluating a partial bundle would
        leave the untouched files at their pristine contents while the edited
        ones move on, which is exactly how a whole experiment produced nothing
        but ``too many arguments to function call`` at the call site of a
        signature the model never saw.
        """
        attempts = self.max_refine_attempts if max_refine_attempts is None else max_refine_attempts
        child_files, problems = self._validate(_bundle_from_arg(kernel_code))
        if problems:
            return False, kernel_code, "Malformed or out-of-scope bundle: " + "; ".join(problems[:3])
        from triton_kernel_agent.platform.ascend_c import merge_bundle

        if parent_kernel:
            parent = merge_bundle(_bundle_from_arg(parent_kernel), child_files)
        else:
            self.logger.warning(
                "no parent bundle supplied; verifying the reply standalone "
                "(unchanged files will be taken from the pristine template)"
            )
            parent = render_bundle(child_files)

        files, _ = parse_bundle(parent)
        structural = bundle_sanity(parent, files)
        if structural:
            result = EvalResult(
                bundle_id(parent), parent,
                StepResult("prebuild", False, 0.0,
                           "structural pre-check failed:\n" + "\n".join(structural)),
                StepResult("correctness", False, 0.0, "skipped", skipped=True),
                StepResult("benchmark", False, 0.0, "skipped", skipped=True),
                error="structural_error",
            )
        else:
            result = self.evaluator.evaluate(parent)
        error = self._describe_failure(result)
        attempt = 0
        while not result.verified and attempt < max(0, attempts):
            attempt += 1
            self.logger.info(
                f"Candidate {result.variant_id} failed the official gate; "
                f"repair attempt {attempt}/{attempts}"
            )
            repaired = self._repair(parent, error, problem_description, attempt)
            if repaired is None:
                break
            files, _ = parse_bundle(repaired)
            structural = bundle_sanity(parent, files)
            parent = repaired
            if structural:
                result = EvalResult(
                    bundle_id(parent), parent,
                    StepResult("prebuild", False, 0.0,
                               "structural pre-check failed:\n" + "\n".join(structural)),
                    StepResult("correctness", False, 0.0, "skipped", skipped=True),
                    StepResult("benchmark", False, 0.0, "skipped", skipped=True),
                    error="structural_error",
                )
            else:
                result = self.evaluator.evaluate(repaired)
            error = self._describe_failure(result)

        if result.verified:
            return True, parent, ""
        return False, parent, error

    @staticmethod
    def _describe_failure(result: EvalResult) -> str:
        parts = [f"variant={result.variant_id}"]
        if not result.build.ok:
            name = result.build.name
            if name == "prebuild" and "structural pre-check" in result.build.output:
                parts.append("PHASE: structural pre-check (your reply was incomplete)")
            else:
                parts.append("PHASE: compile/link (scripts/build.sh)")
            parts.append(result.build.output[-3000:])
        elif not result.correctness.ok:
            parts.append("PHASE: official correctness + robustness gate")
            parts.append(result.correctness.output[-3000:])
        elif not result.benchmark.ok:
            parts.append("PHASE: official public benchmark")
            parts.append(result.benchmark.output[-2000:])
        else:
            parts.append("unknown failure")
        return "\n".join(parts)

    def _repair(self, bundle: str, error: str, problem_description: str,
                attempt: int) -> str | None:
        prompt = (
            "You emitted an edited Ascend C source bundle and it was rejected.\n\n"
            "## TASK CONTEXT\n" + _clamp(problem_description, 6000) + "\n\n"
            "## YOUR PREVIOUS EMISSION (multi-file bundle)\n" + _clamp(bundle, 60000) + "\n\n"
            "## REJECTION OUTPUT (attempt %d)\n```\n%s\n```\n\n"
            "## HOW TO FIX THIS\n"
            "- If the rejection lists members/files you never declared, your "
            "previous reply dropped part of a file. Re-emit that file COMPLETE, "
            "from its first line to its last closing brace, including every "
            "member declaration and every helper method.\n"
            "- If it is a compiler error, fix exactly that line.\n"
            "- Never write '...' or 'unchanged'; every emitted file must be "
            "complete and byte-exact valid C++.\n"
            "- Do not redesign the algorithm; keep the declared entry points, the "
            "tiling struct ABI, the launch signature and the output layout.\n\n"
            "Reply with the complete changed files in this exact format:\n"
            "<<<FILE solution/op_kernel/rsm_solution_core.h>>>\n"
            "...full file content...\n"
            "<<<ENDFILE>>>\n"
            "Emit every file you touched, and nothing else. No prose, no markdown fences.\n"
            % (attempt, _clamp(error, 6000))
        )
        try:
            text = self._call_llm([{"role": "user", "content": prompt}], max_tokens=24576)
        except Exception as exc:
            self.logger.warning(f"repair LLM call failed: {exc}")
            return None
        try:
            self.workdir.mkdir(parents=True, exist_ok=True)
            (self.workdir / f"repair_{attempt}_reply.txt").write_text(
                text or "", encoding="utf-8")
        except OSError:
            pass
        files, problems = parse_bundle(text)
        if not files:
            self.logger.warning(f"repair reply carried no bundle: {problems[:2]}")
            return None
        allowed = set(self.evaluator.config.writable_files)
        filtered = {p: b for p, b in files.items() if p in allowed}
        dropped = sorted(set(files) - allowed)
        if dropped:
            self.logger.warning(f"repair reply touched frozen files {dropped}; dropped")
        if not filtered:
            return None
        from triton_kernel_agent.platform.ascend_c import merge_bundle

        return merge_bundle(bundle, filtered)


def _clamp(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


# ---------------------------------------------------------------------------
# Worker-level: worker runner
# ---------------------------------------------------------------------------


class AscendCWorkerRunner(WorkerRunner):
    """Spawns ``OptimizationWorker`` processes wired to the Ascend C seams.

    Every worker evaluates through the same shared :class:`VariantEvaluator`
    lock (a mkdir lock inside the work root), so the single NPU card is never
    oversubscribed even though several workers run concurrently.  Concurrency
    is what lets one worker's LLM call overlap another worker's device run.
    """

    def __init__(self, log_dir: Path | str | None = None,
                 logger: logging.Logger | None = None,
                 benchmark_lock: Any | None = None,
                 profiling_semaphore: Any | None = None,
                 openai_model: str = "deepseek-chat",
                 high_reasoning_effort: bool = False,
                 bottleneck_override: str | None = None,
                 worker_kwargs: dict[str, Any] | None = None,
                 gpu_ids: list[int] | None = None,
                 gpu_locks: dict[int, Any] | None = None,
                 **_ignored: Any) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.logger = logger or logging.getLogger(__name__)
        self.benchmark_lock = benchmark_lock
        self.profiling_semaphore = profiling_semaphore
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        # Keep the *reference*: ``OptimizationManager._resolve_platform`` writes
        # the worker-level ``platform_config`` key into this very dict AFTER the
        # worker runner has been constructed, so a copy would lose it.
        self.worker_kwargs = worker_kwargs if worker_kwargs is not None else {}
        self.gpu_ids = list(gpu_ids) if gpu_ids else [0]

    def run_workers(self, candidates: list[dict[str, Any]], round_num: int,
                    problem_file: Path, test_code: list[str], pytorch_baseline: float,
                    shared_history: list[dict], shared_reflexions: list[dict]
                    ) -> list[dict[str, Any]]:
        result_queue: Any = mp.Queue()
        workers: list[mp.Process] = []
        for index, candidate in enumerate(candidates):
            workdir = self.log_dir / "workers" / f"w{index}" / f"r{round_num}"
            workdir.mkdir(parents=True, exist_ok=True)
            worker_model = candidate.get("openai_model") or self.openai_model
            args = (
                index,
                candidate["parent"].kernel_code,
                candidate["parent"].metrics.time_ms,
                candidate["parent"].program_id,
                problem_file,
                test_code,
                workdir,
                workdir / "logs",
                result_queue,
                pytorch_baseline,
                candidate["bottleneck_id"],
                worker_model,
                self.high_reasoning_effort,
                self.bottleneck_override,
                self.worker_kwargs,
                shared_history,
                shared_reflexions,
                candidate.get("baseline_metrics"),
                self.gpu_ids[index % len(self.gpu_ids)],
            )
            process = mp.Process(target=_ascend_c_worker_process, args=args)
            process.start()
            workers.append(process)

        import queue as _queue

        deadline = time.time() + int(self.worker_kwargs.get("worker_timeout", 7200))
        results: list[dict[str, Any]] = []
        remaining = list(workers)
        while remaining and time.time() < deadline:
            while True:
                try:
                    results.append(result_queue.get_nowait())
                except _queue.Empty:
                    break
                except Exception:
                    break
            still_alive = []
            for process in remaining:
                process.join(timeout=0.5)
                if process.is_alive():
                    still_alive.append(process)
                else:
                    process.close()
            remaining = still_alive
        for process in remaining:
            self.logger.warning(f"Worker pid {process.pid} timed out; terminating")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            process.close()
        while True:
            try:
                results.append(result_queue.get_nowait())
            except _queue.Empty:
                break
            except Exception:
                break
        result_queue.close()
        try:
            result_queue.join_thread()
        except Exception:
            pass
        ok = sum(1 for r in results if r.get("success"))
        for r in results:
            if not r.get("success"):
                self.logger.error(
                    f"worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    self.logger.error(f"worker {r.get('worker_id')} traceback:\n"
                                      f"{r.get('traceback')}")
        self.logger.info(
            f"Round {round_num}: {ok}/{len(candidates)} Ascend C workers succeeded "
            f"({len(results)} results)"
        )
        return results


def _ascend_c_worker_process(worker_id: int, kernel_code: str, known_time: float,
                             parent_id: str, problem_file: Path, test_code: list[str],
                             workdir: Path, log_dir: Path, result_queue: Any,
                             pytorch_baseline: float, bottleneck_id: int,
                             openai_model: str, high_reasoning_effort: bool,
                             bottleneck_override: str | None, worker_kwargs: dict,
                             prior_history: list[dict], prior_reflexions: list[dict],
                             baseline_metrics: dict[str, Any] | None,
                             device_id: int) -> None:
    """Worker entry point: one LLM rewrite of the Ascend C source bundle."""
    import os
    import sys

    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker

        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        worker = OptimizationWorker(
            worker_id=worker_id,
            workdir=workdir,
            log_dir=log_dir,
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            bottleneck_id=bottleneck_id,
            benchmark_lock=None,
            profiling_semaphore=None,
            pytorch_baseline_time=pytorch_baseline,
            bottleneck_override=bottleneck_override,
            prior_history=prior_history,
            prior_reflexions=prior_reflexions,
            **worker_kwargs,
        )
        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,
            baseline_metrics=baseline_metrics,
        )
        # The measured contest score lives in the archived evaluation record
        # (the worker's best program id IS the variant id).
        variant_id = metrics.get("best_ptx_hash")
        g_ref = worst = None
        failure = metrics.get("error") or metrics.get("failure_reason")
        if variant_id:
            record = (Path(os.environ.get("ASCENDC_RSM_WORK", ".")) / "variants"
                      / str(variant_id) / "result.json")
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
                g_ref = data.get("g_ref_public")
                worst = data.get("worst_case")
                if not data.get("correctness", {}).get("ok"):
                    failure = failure or data.get("error") or "gate failed"
            except (OSError, json.JSONDecodeError):
                pass
        result_queue.put({
            "success": success,
            "worker_id": worker_id,
            "kernel_code": best_kernel,
            "time_ms": metrics.get("best_time_ms", float("inf")),
            "parent_id": parent_id,
            "openai_model": openai_model,
            "ptx_hash": variant_id,
            "g_ref_public": g_ref,
            "worst_case_speedup": worst,
            "attempt": metrics.get("last_attempt"),
            "reflexion": metrics.get("last_reflexion"),
            # Without this the manager can only log "worker N failed: None",
            # which is indistinguishable from a crash and hides the one piece of
            # information worth acting on (compile error vs correctness failure
            # vs "the model produced no usable bundle").
            "error": None if success else (
                failure or f"no verified candidate (last variant={variant_id}, "
                f"see {workdir}/logs/opt_worker_{worker_id}.log)"
            ),
        })
    except Exception as exc:  # pragma: no cover - reported to the manager
        result_queue.put({
            "success": False,
            "worker_id": worker_id,
            "openai_model": openai_model,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Worker-level: hardware specs / profiler / roofline
# ---------------------------------------------------------------------------


class AscendCSpecsProvider(AcceleratorSpecsProvider):
    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        return dict(_ASCEND_910B3_SPECS)


class AscendCKernelProfiler(KernelProfilerBase):
    """Ascend exposes no NCU counter set: always report ``None`` (timing-only)."""

    def __init__(self, logger: logging.Logger | None = None, **_ignored: Any) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def profile_kernel(self, kernel_file: Path, problem_file: Path, round_num: int,
                       max_retries: int = 2) -> Any | None:
        self._logger.info(
            f"[{round_num}] ascend_c: no hardware counter profiler; "
            "the per-case official timing table is the feedback channel"
        )
        return None


class AscendCRooflineAnalyzer(RooflineAnalyzerBase):
    """No SOL counters: never stop early on roofline."""

    def analyze(self, ncu_metrics: dict[str, Any]) -> RooflineResult:
        return RooflineResult(
            bottleneck="unknown",
            warnings=["ascend_c: no NCU; time-based objective only"],
        )

    def should_stop(self, result: RooflineResult) -> tuple[bool, str]:
        return False, "ascend_c: roofline early-stop disabled"

    def reset_history(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Worker-level: bottleneck analyzer (LLM, source-level)
# ---------------------------------------------------------------------------


_BOTTLENECK_PROMPT = """\
You are a performance engineer for hand-written **Ascend C** kernels on the
Ascend 910B3 (40 AI Cores, 1600 GB/s HBM, AI Core vector unit).

A hardware profiler is NOT available on this target. Instead you get:
1. the complete current kernel sources (multi-file bundle),
2. the official per-case timing table measured against the shipped baseline,
3. the hardware model.

## TARGET HARDWARE
{hardware}

## CURRENT MEASURED TIMING TABLE (official harness, vs shipped baseline)
{measurement}

## CURRENT SOURCES
{bundle}

## TASK
Identify the {num_bottlenecks} most promising **performance** bottleneck direction(s) for this
kernel, ranked. For each one give:
- category: one of `memory`, `compute`, `underutilized`, `launch_overhead`,
  `parallelism`, `scalar_bound`, `latency`
- summary: one line
- reasoning: cite the concrete numbers from the timing table (which cases lose
  the most, and what that shape implies about the kernel)
- root_causes: up to {num_causes} entries, each {{"cause": "...", "evidence": [{{"metric": "...", "value": "..."}}]}}
- recommended_fixes: up to {num_fixes} entries, each {{"fix": "...", "rationale": "..."}}

Hard rules you must respect when recommending anything:
- the evaluation compares against a numerically strict FP64 reference; any change
  to the accumulation order or dtype must be justified as *still within tolerance*
- do not propose adding new files (the submission may only edit the existing ones)
- block_dim is capped at 40 (AI Cores); a segment or tiling strategy that uses
  fewer cores wastes the device

Respond with a JSON array only:
[{{"category": "...", "summary": "...", "reasoning": "...", "root_causes": [...], "recommended_fixes": [...]}}]
"""


class AscendCBottleneckAnalyzer(BottleneckAnalyzerBase):
    """LLM bottleneck analysis driven by measured timings + source reading."""

    def __init__(self, logger: logging.Logger | None = None,
                 log_dir: Path | str | None = None,
                 openai_model: str = "deepseek-chat",
                 gpu_name: str | None = None,
                 num_bottlenecks: int = 1,
                 num_causes: int = 2,
                 num_fixes: int = 2,
                 ascend_c: dict[str, Any] | None = None,
                 **_ignored: Any) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else Path(".")
        self._model = openai_model
        self._num_bottlenecks = max(1, int(num_bottlenecks))
        self._num_causes = max(1, int(num_causes))
        self._num_fixes = max(1, int(num_fixes))
        self.roofline = AscendCRooflineAnalyzer()
        self._ascend_c = ascend_c or {}
        self._provider: Any | None = None
        try:
            from utils.providers import get_model_provider

            self._provider = get_model_provider(self._model)
        except Exception as exc:  # pragma: no cover
            self._logger.warning(f"bottleneck analyzer provider unavailable: {exc}")

    def _evaluator(self) -> VariantEvaluator:
        return _make_evaluator(self._ascend_c, self._logger)

    def analyze(self, kernel_code: str, ncu_metrics: dict[str, Any],
                round_num: int = 0, roofline_result: Any | None = None) -> list[Any]:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
            BottleneckResult,
        )

        measurement = self._evaluator().read_live_report() or "(no measurement yet)"
        hardware = json.dumps(_ASCEND_910B3_SPECS, indent=2, sort_keys=True)
        prompt = _BOTTLENECK_PROMPT.format(
            hardware=hardware,
            measurement=_clamp(measurement, 8000),
            bundle=_clamp(kernel_code, 60000),
            num_bottlenecks=self._num_bottlenecks,
            num_causes=self._num_causes,
            num_fixes=self._num_fixes,
        )
        try:
            if self._provider is None:
                raise RuntimeError("no LLM provider")
            text = _llm_call(self._provider, self._model,
                             [{"role": "user", "content": prompt}], self._logger,
                             max_tokens=4096)
        except Exception as exc:
            self._logger.warning(f"[{round_num}] bottleneck analysis LLM call failed: {exc}")
            return []

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            (self._log_dir / f"round{round_num:03d}_ascendc_analysis.txt").write_text(
                text, encoding="utf-8"
            )
        except OSError:
            pass

        payload = _extract_json_array(text)
        if not payload:
            self._logger.warning(f"[{round_num}] could not parse bottleneck JSON")
            return []
        results = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("category"):
                continue
            results.append(BottleneckResult(
                category=str(item.get("category", "unknown")),
                summary=str(item.get("summary", "")),
                reasoning=str(item.get("reasoning", "")),
                root_causes=list(item.get("root_causes") or []),
                recommended_fixes=list(item.get("recommended_fixes") or []),
            ))
        self._logger.info(
            f"[{round_num}] ascend_c analyzer produced {len(results)} bottleneck(s): "
            f"{[r.category for r in results]}"
        )
        return results


def _extract_json_array(text: str) -> list[Any] | None:
    import re

    if not text:
        return None
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


# ---------------------------------------------------------------------------
# Worker-level: live-report prescriber (the "profiling" channel)
# ---------------------------------------------------------------------------

_LIVE_REPORT_PREAMBLE = """\
## MEASURED EVIDENCE (official public harness, this is your real feedback channel)

There is no NCU on Ascend, so use the numbers below: they are produced by the
same harness that scores the submission. `G_ref_public` is the weighted
geometric-mean speedup over the shipped baseline (1.0 = parity, higher is
better) and `speedup` per row is that case's own ratio. A candidate that fails
the correctness gate (or does not compile) scores 0 and is discarded.

"""


class AscendCLiveReportPrescriber(RAGPrescriberBase):
    """Injects the newest real measurement into every optimisation prompt.

    The skeleton reaches for a RAG knowledge base here; we repurpose the same
    seam as a deterministic channel for the live evaluation report plus a short
    Ascend C technique checklist, because that is the information the LLM
    actually needs on this target.
    """

    def __init__(self, logger: logging.Logger | None = None,
                 ascend_c: dict[str, Any] | None = None, **_ignored: Any) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._ascend_c = ascend_c or {}

    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        try:
            report = _make_evaluator(self._ascend_c, self._logger).read_live_report()
        except Exception as exc:  # pragma: no cover
            self._logger.warning(f"live report unavailable: {exc}")
            report = ""
        return {"report": report, "query": query}, {"score": 1.0}

    def build_context(self, opt_node: Any, **_ignored: Any) -> str:
        report = ""
        if isinstance(opt_node, dict):
            report = str(opt_node.get("report") or "")
        body = _LIVE_REPORT_PREAMBLE + (report.strip() or "(no measurement recorded yet)")
        return body + "\n\n" + _ASCEND_C_CHECKLIST


_ASCEND_C_CHECKLIST = """\
## ASCEND C TECHNIQUE CHECKLIST (ordered by expected payoff on this operator)

1. **Core occupancy / work distribution.** `GetBlockNum()` is the launched
   `block_dim`. Today the host sets `block_dim = min(S, 40)`, so any shape with
   fewer than 40 segments (e.g. the `S=1, S=4` cases) runs on a fraction of the
   device. Splitting one segment's rows across several cores (partial
   sum/sum-of-squares + a deterministic second phase) is usually the largest
   single win, as long as the reduction order stays reproducible.
2. **Work per core.** A grid-stride loop over segments costs one `DataCopyPad`
   round trip per tile; hoisting the two-pass structure (max, then sum) and
   keeping partials in UB (`TBuf`) avoids re-reading `score` from GM.
3. **DMA efficiency.** `DataCopyPad` with 16-bit elements and D not a multiple
   of 8/16 forces element-wise fallback. Prefer `DataCopyPad` with a 32B-aligned
   block and cast to fp32 once in UB; avoid an extra GM round trip per row.
4. **Scalar bottleneck.** `GetValue()` on a UB tensor is a scalar read that
   serialises the vector pipe: keep per-row scalars in registers/local
   variables, batch where possible, and use `ReduceMax`/`ReduceSum` instead of
   per-element loops.
5. **Vector instruction choice.** `Exp`, `Ln`, `Rsqrt` are 1/8-rate vs `Mul`;
   `Muls`/`Adds` are cheaper than broadcasting. Fuse the weight multiply with
   the accumulate to halve the vector work (`Muls` then `Add`).
6. **Inter-core reduction.** If you split a segment, either (a) run a second
   kernel phase that reads per-core partials from `workspace` (the runner
   gives you up to 4 MiB and a `workspace` pointer), or (b) use
   `SyncAll`/`CrossCoreSetFlag`/`CrossCoreWaitFlag`, or (c) use atomics
   (`SetAtomicAdd<float>`) — but note float atomics are non-deterministic and
   the correctness gate is a strict FP64 reference, so prefer (a)/(b).
7. **Numerical guard rails.** The gate compares `mean`, `rstd`, `logsumexp`
   against an FP64 reference with tight tolerances. When you parallelise a
   reduction: keep the *same* fp32 accumulation scheme per partial, add the
   partials in fp32 in a fixed order (or in fp64), and keep the
   max-subtraction (`score - max`) for stability.
"""


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def register(registry: Any) -> None:
    """Register every ``ascend_c`` component on *registry*."""
    from triton_kernel_agent.platform.registry import registry as default_registry

    target = registry if registry is not None else default_registry
    for component, factory in {
        "verifier": AscendCVerifier,
        "benchmarker": AscendCBenchmarker,
        "worker_runner": AscendCWorkerRunner,
        "worker_benchmarker": AscendCWorkerBenchmark,
        "verification_worker": AscendCVerificationWorker,
        "specs_provider": AscendCSpecsProvider,
        "profiler": AscendCKernelProfiler,
        "roofline_analyzer": AscendCRooflineAnalyzer,
        "bottleneck_analyzer": AscendCBottleneckAnalyzer,
        "rag_prescriber": AscendCLiveReportPrescriber,
    }.items():
        target.register(component, "ascend_c", factory)


__all__ = [
    "EDITABLE_FILES",
    "AscendCBenchmarker",
    "AscendCBottleneckAnalyzer",
    "AscendCKernelProfiler",
    "AscendCLiveReportPrescriber",
    "AscendCRooflineAnalyzer",
    "AscendCSpecsProvider",
    "AscendCVerificationWorker",
    "AscendCVerifier",
    "AscendCWorkerBenchmark",
    "AscendCWorkerRunner",
    "bundle_id",
    "register",
]
