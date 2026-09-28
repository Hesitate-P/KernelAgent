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

"""Ascend NPU platform components (torch_npu + triton-ascend).

The verification / benchmarking / worker-orchestration components reuse the
NVIDIA implementations (they only need the ``target_platform`` knob), while
profiling degrades gracefully: there is no NCU on Ascend, so the profiler
reports ``None`` and the orchestrator falls back to timing-only feedback
driven by LLM source-level analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .interfaces import (
    AcceleratorSpecsProvider,
    KernelProfilerBase,
    RooflineAnalyzerBase,
    RooflineResult,
)
from .nvidia import NvidiaBenchmarker, NvidiaVerifier, NvidiaWorkerRunner

# Approximate public specifications for the Ascend 910B3 (Atlas 800T A2 class).
# Values are engineering estimates used for LLM bottleneck reasoning only.
_ASCEND_910B3_SPECS: dict[str, Any] = {
    "name": "Ascend 910B3",
    "architecture": "Ascend Da Vinci",
    "peak_fp16_tflops": 313.0,
    "peak_bf16_tflops": 313.0,
    "peak_fp32_tflops": 78.0,
    "peak_memory_bw_gbps": 1600,
    "sm_count": 48,  # AI Core count (Cube+Vector pairs)
    "ai_core_count": 48,
    "max_threads_per_sm": 1024,
    "l2_cache_mb": 32,
    "memory_gb": 64,
    "memory_type": "HBM2e",
    "form_factor": "SXM",
    "tdp_w": 400,
    "notes": "Cube unit handles matmul (tl.dot); Vector unit handles element-wise. "
    "Approximate specs for LLM reasoning; not official datasheet values.",
}


class AscendVerifier(NvidiaVerifier):
    """Verifies kernels on Ascend NPU via ``VerificationWorker``."""

    def __init__(self, log_dir: Path, logger: logging.Logger, target_platform: str = "npu") -> None:
        super().__init__(log_dir, logger, target_platform=target_platform)


class AscendBenchmarker(NvidiaBenchmarker):
    """Benchmarks kernels on Ascend NPU (torch.npu events / do_bench)."""

    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        warmup: int = 25,
        repeat: int = 100,
        target_platform: str = "npu",
    ) -> None:
        super().__init__(
            log_dir,
            logger,
            benchmark_lock,
            warmup=warmup,
            repeat=repeat,
            target_platform=target_platform,
        )


class AscendWorkerRunner(NvidiaWorkerRunner):
    """Spawns worker processes pinned via ``ASCEND_RT_VISIBLE_DEVICES``."""


class AscendAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Returns Ascend 910B series hardware specifications."""

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        if device_name and "910" in device_name and "a2" in device_name.lower():
            # "Ascend910B4" style names from npu-smi on Atlas 800T A2
            return _ASCEND_910B3_SPECS
        return _ASCEND_910B3_SPECS


class AscendKernelProfiler(KernelProfilerBase):
    """No NCU on Ascend: always reports ``None`` to trigger timing-only mode."""

    def __init__(self, logger: logging.Logger | None = None, **kwargs: Any) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        self._logger.info(
            f"[{round_num}] Ascend platform: no NCU available, "
            "using timing-only optimization feedback"
        )
        return None


class AscendRooflineAnalyzer(RooflineAnalyzerBase):
    """Roofline placeholder without NCU SOL metrics.

    Always reports ``unknown`` bottleneck with full headroom so the
    optimization loop keeps iterating on benchmark feedback alone.
    """

    def analyze(self, ncu_metrics: dict[str, Any]) -> RooflineResult:
        return RooflineResult(
            efficiency_pct=0.0,
            compute_sol_pct=0.0,
            memory_sol_pct=0.0,
            bottleneck="unknown",
            at_roofline=False,
            headroom_pct=100.0,
            uses_tensor_cores=False,
            warnings=["Ascend: NCU SOL metrics unavailable (timing-only mode)"],
        )

    def should_stop(self, result: RooflineResult) -> tuple[bool, str]:
        # Never terminate early on roofline; convergence checks rely on
        # benchmark timings handled elsewhere.
        return False, "timing-only mode: roofline stop disabled"

    def reset_history(self) -> None:
        return None
