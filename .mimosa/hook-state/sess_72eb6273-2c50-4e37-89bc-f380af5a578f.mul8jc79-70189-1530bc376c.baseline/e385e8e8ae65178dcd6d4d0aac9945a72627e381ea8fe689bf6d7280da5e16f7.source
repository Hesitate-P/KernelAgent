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

"""Ascend C platform core: source-bundle handling + official-evaluation driver.

This module carries the *physical* half of the Ascend C plugin ("ascend_c")
that lets the stock KernelAgent skeleton optimise hand-written **Ascend C**
kernels instead of generated Triton kernels:

* :func:`parse_bundle` / :func:`render_bundle` — an Ascend C "kernel" is not a
  single Python file but a *set* of editable source files.  We serialise that
  set into one text blob (the "bundle") so the whole existing plumbing
  (``ProgramEntry``, the JSON database, PTX-style dedup, beam search history)
  keeps working unchanged.
* :class:`RepoSlot` — one working copy of the contest repository that is
  materialised with a bundle, compiled, and evaluated.
* :class:`VariantEvaluator` — runs the *official* public evaluation
  (``build.sh`` -> ``run_correctness.py`` -> ``player_benchmark.py``) and turns
  it into the scalar objective the skeleton optimises.
* :func:`write_live_report` — the profiling substitute: Ascend has no NCU, so
  the "profile" handed back to the LLM is the per-case timing table plus the
  compiler / verifier logs from the most recent real evaluation.

Objective mapping
-----------------
``G_ref_public`` is a *speedup* (higher is better) while every skeleton
component (beam search, divergence checks, "time_ms") is minimisation-based.
The official manifest carries fixed normalised weights ``w_i``, so

    geomean_w(solution_ms) = exp(sum_i w_i * log s_i) = B / G_ref_public

with ``B`` constant across variants.  Minimising that geometric mean is
therefore *exactly* equivalent to maximising ``G_ref_public``, and it is a real
duration in milliseconds, so the beam search and the reflexion prompts stay
meaningful without any display hacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Source bundles
# ---------------------------------------------------------------------------

#: Files the contest allows a submission to edit, relative to the repo root.
EDITABLE_FILES: tuple[str, ...] = (
    "solution/CMakeLists.txt",
    "solution/op_host/rsm_solution_host.cpp",
    "solution/op_host/rsm_solution_host.h",
    "solution/op_kernel/rsm_solution.cpp",
    "solution/op_kernel/rsm_solution_core.h",
    "solution/op_kernel/rsm_solution_tiling.h",
    "solution/runner/main.cpp",
)

FILE_OPEN = "<<<FILE {path}>>>"
FILE_CLOSE = "<<<ENDFILE>>>"

_BUNDLE_RE = re.compile(
    r"<<<FILE\s+([A-Za-z0-9_./\-]+)\s*>>>[ \t]*\r?\n(.*?)\r?\n?<<<ENDFILE>>>",
    re.DOTALL,
)


def render_bundle(files: dict[str, str], order: Iterable[str] | None = None) -> str:
    """Serialise ``{relative_path: content}`` into the canonical bundle text."""
    paths = [p for p in (order or EDITABLE_FILES) if p in files]
    paths += [p for p in sorted(files) if p not in paths]
    chunks = []
    for path in paths:
        body = files[path].replace("\r\n", "\n").rstrip("\n")
        chunks.append(FILE_OPEN.format(path=path) + "\n" + body + "\n" + FILE_CLOSE)
    return "\n".join(chunks) + "\n"


def parse_bundle(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse bundle *text* into ``(files, problems)``.

    ``problems`` is a list of human-readable rejection reasons; an empty list
    means the bundle is structurally acceptable.  Unknown paths are rejected
    because the contest forbids adding files.
    """
    files: dict[str, str] = {}
    problems: list[str] = []
    for match in _BUNDLE_RE.finditer(text):
        path, body = match.group(1), match.group(2)
        if path not in EDITABLE_FILES:
            problems.append(
                f"file {path!r} is not one of the editable files "
                f"({', '.join(EDITABLE_FILES)})"
            )
            continue
        if not body.strip():
            problems.append(f"file {path!r} is empty")
            continue
        files[path] = body
    if not files:
        problems.append(
            "no <<<FILE path>>>...<<<ENDFILE>>> block found - the response must "
            "contain at least one complete editable file"
        )
    return files, problems


def bundle_id(text: str) -> str:
    """Stable short id of the *canonical* form of a bundle (dedup key)."""
    files, _ = parse_bundle(text)
    if files:
        canonical = render_bundle(files)
    else:
        canonical = text
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def merge_bundle(parent_text: str, child_files: dict[str, str]) -> str:
    """Apply a (possibly partial) child bundle on top of *parent_text*.

    Agents are asked to emit only the files they change; unchanged files are
    inherited from the parent.
    """
    parent_files, _ = parse_bundle(parent_text)
    merged = dict(parent_files)
    merged.update(child_files)
    return render_bundle(merged)


#: Files of the optional "seed harness" overlay (see :func:`seed_overlay`).
SEED_DIRNAME = "ascend_c_rsm"

#: A kernel that already receives ``workspace`` and the tiling object through
#: ``Init`` has the plumbing the overlay exists to add.
_HARNESS_INIT_RE = re.compile(r"Init\s*\([^)]*GM_ADDR\s+workspace", re.DOTALL)


def harness_present(template_dir: Path | str) -> bool:
    """True when the template's kernel already takes ``(workspace, tiling)``.

    The overlay in :func:`seed_overlay` is a *mechanical* upgrade of the
    shipped, primitive kernel.  Once a template is a copy of a source tree
    that has already been developed (the contest repo itself, or a colleague's
    best hand-written version), applying the overlay would *downgrade* it back
    to the primitive algorithm - silently, and exactly at the moment the
    search has the most to gain from a strong parent.  Checking the call
    signature is enough to tell the two apart.
    """
    core = Path(template_dir) / "solution/op_kernel/rsm_solution_core.h"
    try:
        text = core.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_HARNESS_INIT_RE.search(text))


def seed_overlay(template_dir: Path | str, seed_dir: Path | str | None = None) -> dict[str, str]:
    """Load an optional harness overlay applied to the pristine sources.

    The shipped ``solution/`` is a correct but deliberately primitive kernel:
    among other things the ``workspace`` pointer handed to the kernel entry is
    dropped on the floor (``(void)workspace;``) and the tiling struct's fields
    are passed as loose scalars.  Almost every interesting optimisation of this
    operator (splitting a segment across cores, batching short segments) needs
    that scratch buffer and needs the launch parameters in one place, so a
    *mechanical, algorithm-preserving* harness change unlocks the search space
    without itself changing the numerics.

    The overlay is a directory of complete files (paths relative to the repo
    root) that replaces the pristine ones in the starting bundle.  It lives in
    the KernelAgent checkout (``examples/seeds/<name>/``) so the contest
    template on the cluster stays byte-identical to the shipped problem.

    The overlay is skipped when the template already carries the harness (see
    :func:`harness_present`), because then it could only make the starting
    candidate *worse*.
    """
    explicit = seed_dir is not None
    if seed_dir is None:
        candidate = Path(__file__).resolve().parents[2] / "examples" / "seeds" / SEED_DIRNAME
        if not candidate.is_dir():
            return {}
        seed_dir = candidate
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return {}
    if harness_present(template_dir):
        logger = logging.getLogger(__name__)
        if explicit:
            logger.warning(
                "[ascend_c] the template already carries the workspace/tiling "
                "harness, but an overlay was requested explicitly: the overlay "
                "files will REPLACE the template's own versions"
            )
        else:
            logger.info(
                "[ascend_c] seed harness overlay skipped: the template already "
                "carries the workspace/tiling plumbing (nothing to add)"
            )
            return {}
    overlay: dict[str, str] = {}
    for relative in EDITABLE_FILES:
        path = seed_dir / relative
        if path.is_file():
            overlay[relative] = path.read_text(encoding="utf-8")
    return overlay


def starting_files(template_dir: Path | str,
                   seed_dir: Path | str | None = None) -> dict[str, str]:
    """The pristine editable files with the optional seed overlay applied."""
    template_dir = Path(template_dir)
    files = {
        relative: (template_dir / relative).read_text(encoding="utf-8")
        for relative in EDITABLE_FILES
    }
    files.update(seed_overlay(template_dir, seed_dir))
    return files


def starting_bundle(template_dir: Path | str,
                    seed_dir: Path | str | None = None) -> str:
    return render_bundle(starting_files(template_dir, seed_dir))


# ---------------------------------------------------------------------------
# Pre-build sanity checks
# ---------------------------------------------------------------------------

#: Identifiers that are not members of the class but still end in ``_``.
_NOT_MEMBERS = frozenset({
    "kernel_operator_h", "gm_", "aclrtlaunch_", "kernel_", "__aicore__", "aicore_",
})

# A *declaration* line: a type-ish prefix, then ``name_``, then ``;`` or
# ``= ...;`` or ``[...]``.  Deliberately line-local ([ \t] instead of \s) and
# conservative, so that an assignment such as ``d_ = d;`` cannot masquerade as
# a declaration of ``d_``.
_DECL_LINE_RE = re.compile(
    r"^[ \t]*(?:const[ \t]+|static[ \t]+|__gm__[ \t]+)*"
    r"[A-Za-z_][\w:<>,\*&\.]*[ \t\*&]([A-Za-z_]\w*_)"
    r"(?:[ \t]*\[[^\]]*\])?[ \t]*(?:=[^;]*)?;",
    re.M,
)
_USE_RE = re.compile(r"\b([A-Za-z_]\w*_)\b")

_ELISION_MARKERS = ("...", "…", "<unchanged", "// unchanged", "/* unchanged",
                    "rest of the file", "same as before")


def _declared_names(text: str) -> set[str]:
    return set(_DECL_LINE_RE.findall(text))


def bundle_sanity(parent_text: str, child_files: dict[str, str]) -> list[str]:
    """Cheap structural checks on an LLM-proposed edit, *before* compiling.

    The dominant failure mode of a whole-file rewrite is a silently dropped
    tail: the model re-emits the file from a partial mental model, keeps using
    the class members, and forgets to re-declare them (the compiler then reports
    a dozen "use of undeclared identifier" errors).  Catching that here turns a
    20-second failed build into an instant, precise rejection, and - because the
    message is specific - gives the repair loop something actionable.

    Checks: size shrink, brace balance, elision markers, and "uses a member the
    parent declared but never re-declares it".
    """
    parent_files, _ = parse_bundle(parent_text)
    problems: list[str] = []
    for path, body in child_files.items():
        original = parent_files.get(path, "")
        if original:
            if len(body) < 0.7 * len(original):
                problems.append(
                    f"{path}: the file shrank from {len(original)} to {len(body)} "
                    "characters - most of the original was dropped"
                )
            child_declared = _declared_names(body)
            used = set(_USE_RE.findall(body))
            missing = [
                name for name in sorted(_declared_names(original))
                if name in used and name not in child_declared
            ]
            if missing:
                problems.append(
                    f"{path}: uses {', '.join(missing[:10])} "
                    f"({len(missing)} member(s)) but never declares them - the "
                    "file is incomplete; every member you use must be declared "
                    "in the class body"
                )
        for marker in _ELISION_MARKERS:
            if marker in body and marker not in original:
                # A marker the *parent* already carries is a legitimate part
                # of the shipped source (px-f1's core.h has "…" inside a
                # Chinese comment and main.cpp uses the C variadic macro
                # ``RSM_LAUNCH(...)``) - only markers the child *introduces*
                # count as evidence of an elision/truncation.
                problems.append(f"{path}: contains the elision marker {marker!r}; "
                                "the full file content is required")
        if body.count("{") != body.count("}") and (
            not original or original.count("{") != original.count("}")
        ):
            # Same rule for braces: the parent's own imbalance (e.g. braces
            # inside string literals or comments) must not condemn a child
            # that merely inherited it.
            problems.append(
                f"{path}: unbalanced braces, {body.count('{')} opening vs "
                f"{body.count('}')} closing"
            )
        if original and original.rstrip().endswith("}") and not body.rstrip().endswith("}"):
            problems.append(f"{path}: does not close with a brace like the original")
    return problems


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AscendCConfig:
    """Runtime configuration for the Ascend C plugin (all env-overridable)."""

    #: Pristine copy of the contest repository used as the materialisation source.
    template_dir: Path
    #: Root that holds the build slot, archived variants, and the live report.
    work_root: Path
    #: ``full`` runs every correctness case; ``fast`` skips the slowest ones.
    correctness_mode: str = "fast"
    #: ``quick`` = 1 benchmark round / 2 groups / 5 ms target; ``formal`` =
    #: 4 rounds / 5 groups / 20 ms target (the official protocol).
    benchmark_mode: str = "quick"
    #: Optional explicit case-id list for the correctness gate.
    correctness_cases: tuple[str, ...] = ()
    #: Optional extra perturbation manifest (the "pz" battery) run as an
    #: additional hard gate right after the official correctness suite.  The
    #: judge is known to swap input data, so candidates must also survive the
    #: perturbed suite - in particular the adversarial underflow families
    #: (``distant_center`` / ``low_mass_tail`` with ``gap`` in the FP32
    #: underflow window ~85..110) that historically produced NaN blow-ups.
    pz_manifest: str = ""
    #: ``gap`` = only the underflow-window adversarial cases (default, ~24);
    #: ``all`` = every adversarial case in the pz manifest (~108).
    pz_scope: str = "gap"
    build_timeout: int = 900
    correctness_timeout: int = 600
    benchmark_timeout: int = 1200
    per_case_timeout: float = 30.0
    build_jobs: int = 8
    #: Run a CPU-only compile gate before the device is locked.  Building the
    #: Ascend C kernel needs no NPU (only the CANN/BiSheng toolchain), so a
    #: candidate that does not compile can be rejected - and repaired - on the
    #: login node, where CPU time is free, instead of holding the NPU lock.
    prebuild: bool = False
    prebuild_jobs: int = 4
    #: Files the *agent* may emit this run (a subset of :data:`EDITABLE_FILES`).
    #: The contest allows editing seven files, but the 17-line kernel adapter
    #: ``solution/op_kernel/rsm_solution.cpp`` is where whole-file rewrites keep
    #: breaking the ``Init`` call site (measured: 8/8 failures in one experiment),
    #: so by default the search may write the core header, the host and the
    #: tiling header and must leave the adapter byte-identical.
    writable_files: tuple[str, ...] = EDITABLE_FILES
    #: Skip ``build.sh`` when the source fingerprint is unchanged (cache).
    reuse_build: bool = True

    @classmethod
    def from_env(cls, template_dir: Path | str | None = None,
                 work_root: Path | str | None = None) -> "AscendCConfig":
        env = os.environ
        tpl = template_dir or env.get("ASCENDC_RSM_TEMPLATE")
        work = work_root or env.get("ASCENDC_RSM_WORK")
        if not tpl:
            raise ValueError(
                "ASCENDC_RSM_TEMPLATE is unset: point it at the pristine contest "
                "repository copy that variants are materialised from"
            )
        if not work:
            raise ValueError("ASCENDC_RSM_WORK is unset: point it at the work root")
        cases = tuple(
            c.strip() for c in env.get("ASCENDC_RSM_CORRECTNESS_CASES", "").split(",")
            if c.strip()
        )
        return cls(
            template_dir=Path(tpl),
            work_root=Path(work),
            correctness_mode=env.get("ASCENDC_RSM_CORRECTNESS", "fast"),
            benchmark_mode=env.get("ASCENDC_RSM_BENCHMARK", "quick"),
            correctness_cases=cases,
            pz_manifest=env.get("ASCENDC_RSM_PZ_MANIFEST", "").strip(),
            pz_scope=env.get("ASCENDC_RSM_PZ_SCOPE", "gap").strip() or "gap",
            build_timeout=int(env.get("ASCENDC_RSM_BUILD_TIMEOUT", "900")),
            correctness_timeout=int(env.get("ASCENDC_RSM_CORRECTNESS_TIMEOUT", "600")),
            benchmark_timeout=int(env.get("ASCENDC_RSM_BENCHMARK_TIMEOUT", "1200")),
            per_case_timeout=float(env.get("ASCENDC_RSM_PER_CASE_TIMEOUT", "30")),
            build_jobs=int(env.get("ASCENDC_RSM_BUILD_JOBS", "8")),
            prebuild=env.get("ASCENDC_RSM_PREBUILD", "0") not in ("0", "", "false"),
            prebuild_jobs=int(env.get("ASCENDC_RSM_PREBUILD_JOBS", "4")),
            writable_files=_parse_writable(env),
            reuse_build=env.get("ASCENDC_RSM_REUSE_BUILD", "1") != "0",
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _tail(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


#: The kernel adapter is byte-frozen by default (see AscendCConfig.writable_files).
DEFAULT_WRITABLE: tuple[str, ...] = tuple(
    path for path in EDITABLE_FILES if path != "solution/op_kernel/rsm_solution.cpp"
)


def _parse_writable(env: dict[str, str]) -> tuple[str, ...]:
    raw = env.get("ASCENDC_RSM_WRITABLE", "default").strip()
    if raw in ("", "default"):
        return DEFAULT_WRITABLE
    if raw == "all":
        return EDITABLE_FILES
    wanted = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [path for path in wanted if path not in EDITABLE_FILES]
    if unknown:
        raise ValueError(f"ASCENDC_RSM_WRITABLE lists non-editable paths: {unknown}")
    return wanted or DEFAULT_WRITABLE


def _run(cmd: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None,
         log_path: Path | None = None) -> tuple[int, str, float]:
    """Run *cmd* in *cwd*, returning ``(returncode, combined_output, seconds)``.

    ``returncode == -1`` means the timeout fired (or the process could not be
    started); the output captured so far is still returned so the LLM can see
    partial compiler output.
    """
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd, cwd=str(cwd), env=child_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout,
        )
        output, code = completed.stdout, completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n[ascend_c] TIMEOUT after {timeout:.0f}s: {' '.join(cmd)}\n"
        code = -1
    except Exception as exc:  # pragma: no cover - defensive
        output = f"[ascend_c] failed to run {' '.join(cmd)}: {exc!r}\n"
        code = -1
    elapsed = time.monotonic() - started
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding="utf-8", errors="replace")
        except OSError:
            pass
    return code, output, elapsed


# ---------------------------------------------------------------------------
# Cross-process lock for the (single) NPU card
# ---------------------------------------------------------------------------


class DirLock:
    """Atomic mkdir lock with stale-owner recovery.

    The contest grants one NPU card per user and several consumers share the
    quota, so any command that touches the device must be serialised.  One NPU
    costs one card no matter how many worker processes the search spawns, so
    every evaluation in this plugin funnels through this lock.  Recovery works
    even after a killed worker: a lock whose owner pid is gone (or that is
    older than ``stale_after``) is stolen.
    """

    def __init__(self, path: Path | str, logger: logging.Logger | None = None,
                 wait_seconds: float = 3600.0, poll_seconds: float = 2.0,
                 stale_after: float = 5400.0) -> None:
        self.path = Path(path)
        self.logger = logger or logging.getLogger(__name__)
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.stale_after = stale_after
        self._held = False

    def _owner_alive(self) -> bool:
        try:
            payload = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True  # unknown owner: do not steal a fresh lock
        pid = int(payload.get("pid", -1))
        started = float(payload.get("time", 0.0))
        if started and time.time() - started > self.stale_after:
            return False
        if pid > 0 and payload.get("host") == os.uname().nodename:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return True

    def acquire(self) -> None:
        deadline = time.monotonic() + self.wait_seconds
        waited = False
        while True:
            try:
                self.path.mkdir(parents=True, exist_ok=False)
                (self.path / "owner.json").write_text(
                    json.dumps({"pid": os.getpid(), "host": os.uname().nodename,
                                "time": time.time()}),
                    encoding="utf-8",
                )
                self._held = True
                return
            except FileExistsError:
                if not self._owner_alive():
                    self.logger.warning(f"[ascend_c] stealing stale NPU lock {self.path}")
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"timed out waiting for the NPU lock {self.path}")
                if not waited:
                    self.logger.info(f"[ascend_c] waiting for the NPU lock {self.path}")
                    waited = True
                time.sleep(self.poll_seconds)

    def release(self) -> None:
        if self._held:
            shutil.rmtree(self.path, ignore_errors=True)
            self._held = False

    def __enter__(self) -> "DirLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Repository slot
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Outcome of a single official-evaluation phase."""

    name: str
    ok: bool
    seconds: float
    output: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "skipped": self.skipped,
            "output_tail": _tail(self.output, 1500),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepResult":
        return cls(
            name=str(data.get("name", "?")),
            ok=bool(data.get("ok", False)),
            seconds=float(data.get("seconds", 0.0)),
            output=str(data.get("output_tail", "")),
            skipped=bool(data.get("skipped", False)),
        )


@dataclass
class EvalResult:
    """Everything the search loop needs to know about one variant."""

    variant_id: str
    bundle: str
    build: StepResult
    correctness: StepResult
    benchmark: StepResult
    compliance: StepResult | None = None
    g_ref_public: float = 0.0
    worst_case: float = 0.0
    objective_ms: float = float("inf")
    per_case: list[dict[str, Any]] = field(default_factory=list)
    cached: bool = False
    error: str = ""

    @property
    def verified(self) -> bool:
        """Compiles, is a legal submission, and passes the numerical gate."""
        if self.compliance is not None and not self.compliance.ok:
            return False
        return self.build.ok and self.correctness.ok

    @property
    def success(self) -> bool:
        return self.verified and self.benchmark.ok and self.objective_ms != float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "build": self.build.to_dict(),
            "correctness": self.correctness.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "compliance": self.compliance.to_dict() if self.compliance else None,
            "g_ref_public": self.g_ref_public,
            "worst_case": self.worst_case,
            "objective_ms": self.objective_ms,
            "per_case": self.per_case,
            "cached": self.cached,
            "error": self.error,
        }


class CpuCompileGate:
    """CPU-only compile gate (no device, no NPU allocation).

    ``scripts/build.sh`` cannot run off-device because ``scripts/env.sh``
    refuses to continue when ``npu-smi`` is unavailable.  Compiling, however,
    needs nothing from the device - only the CANN/BiSheng toolchain - so this
    gate reproduces the exact environment ``env.sh`` would export (by sourcing
    it and dumping the resulting environment) and then runs the same CMake
    commands as ``solution/build.sh`` in a scratch directory.

    It is a *prefilter*: a failure is a reliable rejection, while a success is
    still re-validated by the official build inside the device job.
    """

    _ENV_KEYS = (
        "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CC", "CXX", "CPLUS_INCLUDE_PATH",
        "ASCEND_HOME_PATH", "ASCEND_INSTALL_ROOT", "RSM_TOOLCHAIN_ROOT",
        "CMAKE_PREFIX_PATH", "ASCEND_TOOLCHAIN_HOME", "ASCEND_OPP_PATH",
    )

    def __init__(self, config: AscendCConfig, logger: logging.Logger,
                 slot_name: str = "shared") -> None:
        self.config = config
        self.logger = logger
        # A *stable* name per worker role: the CMake configure + toolchain
        # precompile step costs minutes, while an incremental rebuild of a
        # changed header costs seconds, so the scratch tree must survive across
        # the search rounds (worker processes are respawned every round).
        self.slot_name = re.sub(r"[^A-Za-z0-9_.-]", "_", slot_name)[:48] or "shared"
        self.dir = config.work_root / "cpu_slots" / self.slot_name
        # Cheap safety net: the slot is per worker role, so contention should
        # never happen, but if a caller reuses a slot concurrently the lock
        # degrades it to "serialised" instead of "corrupted".
        self.lock = DirLock(config.work_root / "locks" / f"cpu_{self.slot_name}.lock",
                            logger, wait_seconds=1800.0, stale_after=1800.0)
        self._env: dict[str, str] | None = None

    def _toolchain_env(self) -> dict[str, str]:
        """Environment as produced by ``scripts/env.sh`` (device check ignored)."""
        if self._env is not None:
            return self._env
        probe = (
            "source scripts/env.sh >/dev/null 2>&1 || true; env -0"
        )
        try:
            completed = subprocess.run(
                ["bash", "-c", probe], cwd=str(self.dir), stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=120,
            )
            raw = completed.stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError) as exc:
            self.logger.warning(f"[ascend_c] could not resolve the build env: {exc}")
            raw = ""
        env: dict[str, str] = {}
        for entry in raw.split("\0"):
            if "=" in entry:
                key, value = entry.split("=", 1)
                env[key] = value
        if not env:
            env = dict(os.environ)
        self._env = env
        return env

    def ensure(self) -> None:
        template = self.config.template_dir
        if (self.dir / "solution/CMakeLists.txt").is_file():
            return
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            template, self.dir,
            ignore=shutil.ignore_patterns(
                "artifacts", "build", ".mimosa", "__pycache__", ".git", "ka-*",
                "*.o", "*.so",
            ),
        )

    def compile(self, files: dict[str, str]) -> StepResult:
        self.lock.acquire()
        try:
            return self._compile_locked(files)
        finally:
            self.lock.release()

    def _compile_locked(self, files: dict[str, str]) -> StepResult:
        self.ensure()
        for path, body in files.items():
            target = self.dir / path
            body = body.replace("\r\n", "\n")
            if not body.endswith("\n"):
                body += "\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")

        env = dict(self._toolchain_env())
        env["TMPDIR"] = os.environ.get("TMPDIR", "/dev/shm")
        env.pop("ASCEND_DEVICE_ID", None)
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

        build_dir = self.dir / "build/solution"
        install_dir = self.dir / "build/solution-install"
        started = time.monotonic()
        output = ""
        for cmd in (
            ["cmake", "-S", "solution", "-B", "build/solution", "-G", "Unix Makefiles",
             "-DSOC_VERSION=Ascend910B3", "-DRUN_MODE=npu", "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_INSTALL_PREFIX={install_dir}",
             f"-DASCEND_CANN_PACKAGE_PATH={env.get('ASCEND_HOME_PATH', '')}"],
            ["cmake", "--build", "build/solution", "-j", str(self.config.prebuild_jobs)],
        ):
            code, out, secs = _run(cmd, self.dir, timeout=self.config.build_timeout,
                                  env=env)
            output += f"$ {' '.join(cmd)}\n{out}\n"
            if code != 0:
                return StepResult("prebuild", False, time.monotonic() - started, output)
        return StepResult("prebuild", True, time.monotonic() - started, output)


class RepoSlot:
    """A single mutable working copy of the contest repo used for evaluation.

    The slot owns ``build/`` and ``artifacts/``.  Because the NPU is a shared,
    serialised resource (see :class:`VariantEvaluator`), one slot is enough:
    every evaluation holds the lock across materialise -> build -> correctness
    -> benchmark, so no two variants are ever in the slot at the same time.
    """

    def __init__(self, config: AscendCConfig, logger: logging.Logger | None = None,
                 name: str = "slot") -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.dir = config.work_root / name
        self._current_id: str | None = None

    # -- materialisation ---------------------------------------------------

    def ensure(self) -> None:
        """Create the slot from the pristine template if it does not exist."""
        if self.dir.is_dir() and (self.dir / "problem.yaml").is_file():
            return
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[ascend_c] materialising repo slot from {self.config.template_dir}")
        shutil.copytree(
            self.config.template_dir, self.dir,
            ignore=shutil.ignore_patterns(
                "artifacts", "build", ".mimosa", "__pycache__", ".git",
                "ka-work", "ka-best", "*.o", "*.so",
            ),
        )
        self._current_id = None

    @property
    def current_id(self) -> str | None:
        return self._current_id

    def materialize(self, files: dict[str, str]) -> bool:
        """Write *files* into the slot; return True when anything changed."""
        self.ensure()
        changed = False
        for path, body in files.items():
            target = self.dir / path
            body = body.replace("\r\n", "\n")
            if not body.endswith("\n"):
                body += "\n"
            try:
                existing = target.read_text(encoding="utf-8")
            except OSError:
                existing = None
            if existing != body:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                changed = True
        return changed

    # -- phases ------------------------------------------------------------

    def build(self, log_path: Path | None = None) -> StepResult:
        env = {
            "TMPDIR": os.environ.get("TMPDIR", "/dev/shm"),
            "RSM_BUILD_JOBS": str(self.config.build_jobs),
        }
        code, output, seconds = _run(
            ["bash", "scripts/build.sh"], self.dir, self.config.build_timeout,
            env=env, log_path=log_path,
        )
        return StepResult("build", code == 0, seconds, output)

    def compliance(self, log_path: Path | None = None) -> tuple[StepResult, dict[str, Any]]:
        """The judge's source-policy scan (``scripts/check.sh``).

        This runs *first* in the official evaluation and is a hard reject: a
        candidate that trips the scanner is an invalid submission no matter how
        fast it is, so the search must never be allowed to keep one.
        """
        code, output, seconds = _run(
            ["python3", "tools/compliance/run_checks.py",
             "--output-json", "artifacts/public_compliance_results.json"],
            self.dir, 300, log_path=log_path,
        )
        report: dict[str, Any] = {}
        result_path = self.dir / "artifacts/public_compliance_results.json"
        if result_path.is_file():
            try:
                report = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
        ok = code == 0 and bool(report.get("passed", code == 0))
        return StepResult("compliance", ok, seconds, output), report

    def correctness_case_ids(self) -> list[str]:
        self.ensure()
        manifest = self.dir / "tools/data/public_cases.json"
        specs = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
        if self.config.correctness_cases:
            wanted = set(self.config.correctness_cases)
            return [c["case_id"] for c in specs if c["case_id"] in wanted]
        if self.config.correctness_mode == "full":
            return [c["case_id"] for c in specs]
        # "fast": everything except the cases whose *input reference* work
        # dominates the runtime (the perf shapes are still covered by the
        # benchmark phase, which aborts on any incorrect result).
        skip_markers = ("adversarial_long_stream", "adversarial_tile_boundaries",
                        "adversarial_sparse_shards", "full_scale_reduction")
        return [
            c["case_id"] for c in specs
            if not c["case_id"].startswith("public_perf_")
            and not any(m in c["case_id"] for m in skip_markers)
        ]

    def correctness(self, log_path: Path | None = None) -> tuple[StepResult, dict[str, Any]]:
        """The official correctness gate: the case suite *and* the runtime probes.

        ``scripts/test.sh --robustness`` (which the judge runs) is two phases:
        the manifest suite (``run_correctness.py``) and the runtime robustness
        harness (``runtime_robustness.py``: repeated launches, address rotation,
        mutated inputs, dynamic shape sequences).  Both must pass, so both are
        run here - the robustness phase is also the runtime anti-specialisation
        probe, and a candidate that only survives the fixed shapes must not be
        allowed to look "verified".
        """
        case_ids = self.correctness_case_ids()
        env = {"TMPDIR": os.environ.get("TMPDIR", "/dev/shm")}
        cmd = ["python3", "tools/correctness/run_correctness.py",
               "--manifest", "tools/data/public_cases.json", "--variant", "solution",
               "--timeout-seconds", str(self.config.per_case_timeout),
               "--output-json", "artifacts/public_correctness_results.json"]
        for case_id in case_ids:
            cmd += ["--case-id", case_id]
        code, output, seconds = _run(cmd, self.dir, self.config.correctness_timeout,
                                     env=env, log_path=log_path)
        report: dict[str, Any] = {}
        result_path = self.dir / "artifacts/public_correctness_results.json"
        if result_path.is_file():
            try:
                report = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
        ok = code == 0 and bool(report.get("passed", False))

        # ---- pz perturbation gate (hard, right after the official suite) ----
        # The judge swaps input data between our runs and the submission; the
        # perturbed "pz" battery replays the public shapes with new seeds and,
        # crucially, adversarial underflow families whose score gaps fall in
        # the FP32 window (~85..110) where a naive 1/z or z/z' blows up to
        # inf/NaN.  Any candidate that fails here is rejected exactly like an
        # official failure - speed on the fixed public data means nothing if
        # the judge's regenerated data kills the kernel.
        if ok and self.config.pz_manifest:
            pz_path = Path(self.config.pz_manifest).expanduser()
            pz_result_path = self.dir / "artifacts/pz_correctness_results.json"
            try:
                pz_specs = json.loads(
                    pz_path.read_text(encoding="utf-8"))["cases"]
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                ok = False
                output += (f"\n\n===== pz gate =====\nunreadable pz manifest "
                           f"{pz_path}: {exc}")
                return (StepResult("correctness", ok, seconds, output),
                        {**report, "pz_gate": {"error": str(exc)}})
            if self.config.pz_scope == "all":
                pz_ids = [c["case_id"] for c in pz_specs if c.get("adversarial")]
            else:  # "gap": underflow-window families only (distant_center /
            # low_mass_tail carry gap=100.0)
                pz_ids = [c["case_id"] for c in pz_specs if "gap" in c]
            if not pz_ids:
                ok = False
                output += "\n\n===== pz gate =====\nno adversarial cases found"
                return (StepResult("correctness", ok, seconds, output),
                        {**report, "pz_gate": {"error": "no pz cases selected"}})
            pz_log = (log_path.with_suffix(".pz.log")
                      if log_path is not None else None)
            pz_cmd = ["python3", "tools/correctness/run_correctness.py",
                      "--manifest", str(pz_path), "--variant", "solution",
                      "--timeout-seconds", str(self.config.per_case_timeout),
                      "--output-json", str(pz_result_path)]
            for case_id in pz_ids:
                pz_cmd += ["--case-id", case_id]
            pz_timeout = min(
                max(300, int(self.config.per_case_timeout * (len(pz_ids) + 2))),
                self.config.correctness_timeout + 600,
            )
            pcode, poutput, pseconds = _run(
                pz_cmd, self.dir, pz_timeout, env=env, log_path=pz_log)
            pz_report: dict[str, Any] = {}
            if pz_result_path.is_file():
                try:
                    pz_report = json.loads(pz_result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pz_report = {}
            pz_ok = pcode == 0 and bool(pz_report.get("passed", False))
            pz_failures = [
                case.get("case_id")
                for variant in pz_report.get("variants", {}).values()
                for case in variant.get("cases", [])
                if not case.get("passed")
            ]
            report["pz_gate"] = {
                "manifest": str(pz_path), "cases": len(pz_ids),
                "scope": self.config.pz_scope, "passed": pz_ok,
                "failures": pz_failures,
            }
            seconds += pseconds
            output += (f"\n\n===== pz gate ({self.config.pz_scope}, "
                       f"{len(pz_ids)} cases) =====\n"
                       + ("PASS" if pz_ok else f"FAILED: {pz_failures}")
                       + "\n" + poutput)
            ok = ok and pz_ok
            if not pz_ok:
                return StepResult("correctness", ok, seconds, output), report

        if ok:
            robust_cmd = [
                "python3", "tools/correctness/runtime_robustness.py",
                "--variant", "solution",
                "--timeout-seconds", str(max(self.config.per_case_timeout, 30.0)),
                "--output-json", "artifacts/runtime_robustness_results.json",
            ]
            robust_log = (log_path.with_suffix(".robustness.log")
                          if log_path is not None else None)
            rcode, routput, rseconds = _run(
                robust_cmd, self.dir,
                max(self.config.correctness_timeout // 2, 120), env=env,
                log_path=robust_log,
            )
            rreport: dict[str, Any] = {}
            rpath = self.dir / "artifacts/runtime_robustness_results.json"
            if rpath.is_file():
                try:
                    rreport = json.loads(rpath.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    rreport = {}
            robust_ok = rcode == 0 and bool(rreport.get("passed", False))
            report["robustness"] = rreport
            seconds += rseconds
            output += "\n\n===== runtime robustness =====\n" + routput
            ok = ok and robust_ok
        return StepResult("correctness", ok, seconds, output), report

    def benchmark(self, log_path: Path | None = None) -> tuple[StepResult, dict[str, Any]]:
        # NOTE: no ``--no-build`` here.  ``player_benchmark.py`` is the only
        # entry point that builds the *baseline* variant, and it caches builds
        # by (source fingerprint, toolchain, installed-binary hash) - so the
        # first run pays for the baseline and every later run reuses it while
        # picking up the new candidate solution.  Skipping the build step would
        # leave ``build/baseline-install`` missing and abort the run.
        cmd = ["python3", "tools/benchmark/player_benchmark.py"]
        if self.config.benchmark_mode != "formal":
            cmd.append("--quick")
        code, output, seconds = _run(
            cmd, self.dir, self.config.benchmark_timeout,
            env={"TMPDIR": os.environ.get("TMPDIR", "/dev/shm"),
                 "RSM_BUILD_JOBS": str(self.config.build_jobs)},
            log_path=log_path,
        )
        report: dict[str, Any] = {}
        result_path = self.dir / "artifacts/public_benchmark.json"
        if result_path.is_file():
            try:
                report = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
        ok = code == 0 and report.get("G_ref_public") is not None
        return StepResult("benchmark", ok, seconds, output), report


def weighted_geomean_ms(report: dict[str, Any]) -> float:
    """``exp(sum_i w_i log solution_ms_i)`` — a duration strictly monotone in 1/G."""
    rows = report.get("cases") or []
    total_weight = 0.0
    acc = 0.0
    for row in rows:
        weight = float(row.get("public_normalized_weight", 0.0))
        ms = float(row.get("solution_ms", 0.0))
        if weight <= 0 or not math.isfinite(ms) or ms <= 0:
            return float("inf")
        total_weight += weight
        acc += weight * math.log(ms)
    if total_weight <= 0:
        return float("inf")
    return math.exp(acc / total_weight)


# ---------------------------------------------------------------------------
# Variant evaluator
# ---------------------------------------------------------------------------


class VariantEvaluator:
    """Evaluates source bundles with the official harness, under one NPU lock.

    ``lock`` is an ``multiprocessing.Lock`` shared by every worker process (the
    manager hands the same per-GPU lock to all of them), which is what makes
    the single shared :class:`RepoSlot` safe and what keeps the NPU from being
    oversubscribed by parallel workers.
    """

    def __init__(self, config: AscendCConfig, lock: Any | None = None,
                 logger: logging.Logger | None = None,
                 slot_name: str = "manager") -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.slot = RepoSlot(config, self.logger)
        self.variants_dir = config.work_root / "variants"
        self.state_dir = config.work_root / "state"
        self.log_dir = config.work_root / "eval_logs"
        #: ``lock`` is an optional extra mutex (the skeleton's per-GPU lock when
        #: the manager supplies one); the mkdir lock is always taken first, so
        #: acquisition order is consistent across all processes.
        self.lock = lock
        self.device_lock = DirLock(
            config.work_root / "locks" / "npu.lock", self.logger,
            stale_after=float(os.environ.get("ASCENDC_RSM_LOCK_STALE", "5400")),
        )
        self.cpu_gate = (CpuCompileGate(config, self.logger, slot_name)
                         if config.prebuild else None)

    # -- variant record ----------------------------------------------------

    def variant_dir(self, variant_id: str) -> Path:
        return self.variants_dir / variant_id

    def cached_result(self, variant_id: str) -> EvalResult | None:
        path = self.variant_dir(variant_id) / "result.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        bundle_path = self.variant_dir(variant_id) / "bundle.txt"
        bundle = bundle_path.read_text(encoding="utf-8") if bundle_path.is_file() else ""
        build = StepResult.from_dict(data.get("build", {}))
        correctness = StepResult.from_dict(data.get("correctness", {}))
        benchmark = StepResult.from_dict(data.get("benchmark", {}))
        compliance = (StepResult.from_dict(data["compliance"])
                      if data.get("compliance") else None)
        result = EvalResult(
            variant_id=variant_id, bundle=bundle, build=build, correctness=correctness,
            benchmark=benchmark, compliance=compliance,
            g_ref_public=float(data.get("g_ref_public", 0.0)),
            worst_case=float(data.get("worst_case", 0.0)),
            objective_ms=float(data.get("objective_ms", float("inf"))),
            per_case=data.get("per_case", []), cached=True,
        )
        return result

    def _archive(self, result: EvalResult, files: dict[str, str]) -> None:
        target = self.variant_dir(result.variant_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "bundle.txt").write_text(result.bundle, encoding="utf-8")
        # Atomic publish: sibling workers read these records concurrently
        # (resume / best_so_far) while this process may be rewriting them.
        tmp = target / f".result.{os.getpid()}.json.tmp"
        tmp.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target / "result.json")
        for path, body in files.items():
            dest = target / "files" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body.replace("\r\n", "\n"), encoding="utf-8")
        # Keep the raw official artefacts (small JSON files) for the report.
        for name in ("artifacts/public_benchmark.json",
                     "artifacts/public_correctness_results.json"):
            src = self.slot.dir / name
            if src.is_file():
                dest = target / Path(name).name
                try:
                    shutil.copy2(src, dest)
                except OSError:
                    pass

    # -- the evaluation itself --------------------------------------------

    def evaluate(self, bundle_text: str, files: dict[str, str] | None = None,
                 force: bool = False) -> EvalResult:
        """Materialise, build, verify, and benchmark one bundle."""
        files = dict(files or {})
        variant_id = bundle_id(bundle_text)
        if not files:
            files, _ = parse_bundle(bundle_text)
        if not force:
            cached = self.cached_result(variant_id)
            if cached is not None:
                self.logger.info(f"[ascend_c] variant {variant_id}: reusing cached evaluation")
                return cached

        # CPU-only compile gate: reject (and let the LLM repair) candidates that
        # cannot compile without ever occupying the device.
        if self.cpu_gate is not None:
            pre = self.cpu_gate.compile(files)
            if not pre.ok:
                self.logger.warning(
                    f"[ascend_c] variant {variant_id}: CPU compile gate failed "
                    f"({pre.seconds:.0f}s); skipping the device job"
                )
                result = EvalResult(
                    variant_id, bundle_text, pre,
                    StepResult("correctness", False, 0.0,
                               "skipped: compile gate failed", skipped=True),
                    StepResult("benchmark", False, 0.0,
                               "skipped: compile gate failed", skipped=True),
                    error="compile_error",
                )
                self._archive(result, files)
                self.write_live_report(result)
                return result

        acquired = False
        self.device_lock.acquire()
        acquired = True
        if self.lock is not None:
            self.lock.acquire()
        try:
            result = self._evaluate_locked(variant_id, bundle_text, files)
        finally:
            if self.lock is not None:
                self.lock.release()
            if acquired:
                self.device_lock.release()
        self._archive(result, files)
        self.write_live_report(result)
        return result

    def _fake_result(self, variant_id: str, bundle_text: str,
                     files: dict[str, str]) -> EvalResult:
        """Development-only synthetic evaluation (``ASCENDC_RSM_FAKE=1``).

        Lets the orchestration plumbing (manager -> workers -> beam search ->
        reflexion -> export) be exercised end to end on a login node with no
        device, before spending scarce NPU time.  The reported G is a
        deterministic function of the bundle contents, so the search mechanics
        are observable but never mistaken for a real measurement.
        """
        seed = int(variant_id, 16)
        g_ref = 1.0 + (seed % 1000) / 1000.0
        objective = 1.0 / g_ref
        fake = StepResult("codegen", True, 0.1, "fake evaluation (ASCENDC_RSM_FAKE=1)")
        return EvalResult(
            variant_id, bundle_text, fake,
            StepResult("correctness", True, 0.1, "fake"),
            StepResult("benchmark", True, 0.1, "fake"),
            g_ref_public=g_ref, worst_case=g_ref * 0.8, objective_ms=objective,
            per_case=[{"case_id": "fake", "weight": 1.0, "baseline_ms": 1.0,
                       "solution_ms": objective, "speedup": g_ref}],
        )

    def _evaluate_locked(self, variant_id: str, bundle_text: str,
                         files: dict[str, str]) -> EvalResult:
        started = time.monotonic()
        self.logger.info(f"[ascend_c] variant {variant_id}: evaluating "
                         f"({len(files)} files)")
        if os.environ.get("ASCENDC_RSM_FAKE"):
            return self._fake_result(variant_id, bundle_text, files)
        self.slot.materialize(files)
        logs = self.log_dir / variant_id
        logs.mkdir(parents=True, exist_ok=True)

        build = self.slot.build(logs / "build.log")
        if not build.ok:
            self.logger.warning(f"[ascend_c] variant {variant_id}: build failed "
                                f"({build.seconds:.0f}s)")
            skipped = StepResult("skipped", False, 0.0, "skipped: build failed",
                                 skipped=True)
            return EvalResult(
                variant_id, bundle_text, build,
                StepResult("correctness", False, 0.0, "skipped: build failed", skipped=True),
                StepResult("benchmark", False, 0.0, "skipped: build failed", skipped=True),
                compliance=skipped, error="compile_error",
            )

        compliance, compliance_report = self.slot.compliance(logs / "compliance.log")
        if not compliance.ok:
            findings = (compliance_report.get("reject_findings") or [])[:5]
            self.logger.warning(
                f"[ascend_c] variant {variant_id}: compliance scan rejected "
                f"{len(compliance_report.get('reject_findings') or [])} finding(s): "
                f"{findings}"
            )
            skipped = StepResult("skipped", False, 0.0, "skipped: illegal submission",
                                 skipped=True)
            return EvalResult(
                variant_id, bundle_text, build, skipped, skipped,
                compliance=compliance, error="compliance_error",
            )

        correctness, report = self.slot.correctness(logs / "correctness.log")
        if not correctness.ok:
            failures = [
                case.get("case_id") for case in
                (report.get("variants", {}).get("solution", {}).get("cases") or [])
                if not case.get("passed")
            ][:8]
            self.logger.warning(f"[ascend_c] variant {variant_id}: correctness failed "
                                f"({correctness.seconds:.0f}s) cases={failures}")
            return EvalResult(
                variant_id, bundle_text, build, correctness,
                StepResult("benchmark", False, 0.0, "skipped: correctness failed", skipped=True),
                compliance=compliance,
                error="correctness_error:" + ",".join(str(f) for f in failures),
            )

        benchmark, bench_report = self.slot.benchmark(logs / "benchmark.log")
        if not benchmark.ok or bench_report.get("G_ref_public") is None:
            self.logger.warning(
                f"[ascend_c] variant {variant_id}: benchmark failed "
                f"({benchmark.seconds:.0f}s)"
            )
            return EvalResult(variant_id, bundle_text, build, correctness, benchmark,
                              compliance=compliance, error="benchmark_error")

        objective = weighted_geomean_ms(bench_report)
        g_ref = float(bench_report.get("G_ref_public", 0.0))
        rows = [
            {
                "case_id": row.get("case_id"),
                "weight": row.get("public_normalized_weight"),
                "baseline_ms": row.get("baseline_ms"),
                "solution_ms": row.get("solution_ms"),
                "speedup": row.get("speedup"),
            }
            for row in bench_report.get("cases", [])
        ]
        self.logger.info(
            f"[ascend_c] variant {variant_id}: G_ref_public={g_ref:.4f} "
            f"worst={bench_report.get('worst_case_speedup', 0):.4f} "
            f"objective={objective:.4f}ms build={build.seconds:.0f}s "
            f"compliance={compliance.seconds:.0f}s "
            f"correctness={correctness.seconds:.0f}s bench={benchmark.seconds:.0f}s "
            f"total={time.monotonic() - started:.0f}s"
        )
        return EvalResult(
            variant_id, bundle_text, build, correctness, benchmark,
            compliance=compliance,
            g_ref_public=g_ref, worst_case=float(bench_report.get("worst_case_speedup", 0.0)),
            objective_ms=objective, per_case=rows,
        )

    # -- the "profiler" substitute ----------------------------------------

    @property
    def live_report_path(self) -> Path:
        return self.state_dir / "live_report.md"

    def write_live_report(self, result: EvalResult) -> None:
        """Publish the newest measurement as markdown (read back by the LLM)."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# LIVE MEASUREMENT REPORT (official public harness)",
            "",
            "## WRITABLE FILES THIS ROUND",
            "Emit ONLY these paths; any other path is rejected before compiling.",
        ]
        lines += [f"- `{path}`" for path in self.config.writable_files]
        frozen = [p for p in EDITABLE_FILES if p not in self.config.writable_files]
        if frozen:
            lines += [
                "",
                "These files are FROZEN (the harness restores them if you touch "
                "them, and a changed call signature will not compile):",
            ]
            lines += [f"- `{path}`" for path in frozen]
        lines += [
            "",
            f"Newest evaluated candidate: `{result.variant_id}`"
            + ("  (cached result)" if result.cached else ""),
            "",
            f"- build: {'ok' if result.build.ok else 'FAILED'} "
            f"({result.build.seconds:.0f}s)",
            f"- compliance scan (source policy): "
            f"{'PASS' if (result.compliance is None or result.compliance.ok) else 'REJECTED'}",
            f"- correctness+robustness gate: "
            f"{'PASS' if result.correctness.ok else 'FAILED'} "
            f"({result.correctness.seconds:.0f}s)",
            f"- public benchmark: {'ok' if result.benchmark.ok else 'not run'} "
            f"({result.benchmark.seconds:.0f}s)",
            f"- **G_ref_public = {result.g_ref_public:.4f}x**  "
            f"(weighted geometric-mean speedup vs the shipped baseline; 1.0 = parity)",
            f"- worst case speedup = {result.worst_case:.4f}x",
        ]
        if result.error:
            lines.append(f"- failure: `{result.error}`")
        if result.per_case:
            lines += [
                "",
                "| case | weight | baseline ms | this candidate ms | speedup |",
                "|---|---|---|---|---|",
            ]
            for row in result.per_case:
                lines.append(
                    "| {case_id} | {weight:.4f} | {baseline_ms:.5f} | "
                    "{solution_ms:.5f} | {speedup:.3f}x |".format(
                        case_id=row["case_id"], weight=float(row["weight"]),
                        baseline_ms=float(row["baseline_ms"]),
                        solution_ms=float(row["solution_ms"]),
                        speedup=float(row["speedup"]),
                    )
                )
        if not result.correctness.ok and result.correctness.output:
            lines += ["", "## correctness gate output (tail)", "```",
                      _tail(result.correctness.output, 2500), "```"]
        elif not result.build.ok and result.build.output:
            lines += ["", "## compiler output (tail)", "```",
                      _tail(result.build.output, 2500), "```"]
        elif not result.benchmark.ok and result.benchmark.output:
            lines += ["", "## benchmark output (tail)", "```",
                      _tail(result.benchmark.output, 1500), "```"]
        text = "\n".join(lines) + "\n"
        # Per-process temp name: several workers finish evaluations around the
        # same time and all refresh this file.
        tmp = self.live_report_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.live_report_path)

    def read_live_report(self) -> str:
        try:
            return self.live_report_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def best_so_far(self) -> dict[str, Any] | None:
        """Best measured variant across the whole work root (audit trail)."""
        best: dict[str, Any] | None = None
        if not self.variants_dir.is_dir():
            return None
        for path in sorted(self.variants_dir.glob("*/result.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            g = float(data.get("g_ref_public", 0.0))
            if not data.get("benchmark", {}).get("ok"):
                continue
            if best is None or g > float(best.get("g_ref_public", 0.0)):
                best = data
        return best


# ---------------------------------------------------------------------------
# Problem statement (the static half of the prompt context)
# ---------------------------------------------------------------------------

_NUMERIC_CONTRACT = """\
## NUMERICAL CONTRACT (this is what the correctness gate enforces)

For every segment `s` (rows `[offsets[s], offsets[s+1])`):

    w_i     = exp(score_i - max(score_s))            (weights, in fp32)
    mean_d  = sum_i w_i * x[i][d] / sum_i w_i        (per output column d)
    var_d   = sum_i w_i * (x[i][d] - mean_d)^2 / sum_i w_i
    rstd_d  = 1 / sqrt(max(var_d, 0) + epsilon)
    logsumexp_s = log(sum_i w_i) + max(score_s)

`score` and `x` are FP16 in global memory; `mean`, `rstd` and `logsumexp` are
FP32 outputs (one row per segment for `mean`/`rstd`, one scalar per segment for
`logsumexp`).

The judge recomputes the same quantities in **FP64** and compares element-wise:

| output | atol | rtol |
|---|---|---|
| mean | 3e-3 (+ a per-input conditioning allowance) | 2^-23 |
| rstd | 0 | 4e-3 |
| logsumexp | 5e-3 | 2^-23 |

So: `mean` and `logsumexp` are absolute-accuracy limited, while `rstd` is
*relative*-accuracy limited (0.4%). `rstd` must additionally be finite and
strictly positive everywhere.

The suite deliberately contains cases that punish common shortcuts:
- `score` values chosen so that `exp(score - max)` underflows to zero for part
  of a segment (`public_underflow_*`, `public_precision_intrinsic_exp_floor_33`,
  `public_precision_correctly_rounded_exp_513`);
- catastrophic cancellation when summing `w * x` of mixed sign
  (`public_regression_signed_cancellation*`, `public_adversarial_signed_order`);
- near-constant columns where `mean` must be accurate to recover a tiny
  variance (`public_precision_constant_requires_accurate_mean_16`,
  `public_large_mean_small_variance`);
- a single outlier with a tiny weight that still shifts the last bits of the
  result (`public_low_weight_outlier*`, `public_large_sparse_perturbation`);
- tile/alignment boundaries at 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127,
  128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
  4095, 4096, 4097 rows, S = 1..8192 segments, D up to 256.

**Consequence:** you may restructure the *order* in which partial sums are
combined only if the result stays inside those tolerances. Splitting work over
cores is fine and encouraged; making each partial sum numerically *worse*
(FP16 accumulation, single-pass variance, dropping the max subtraction,
replacing `Exp` with a lookup) is not.
"""


def build_problem_statement(template_dir: Path | str) -> str:
    """Compose the task description handed to the LLM as ``problem.py``."""
    root = Path(template_dir)
    manifest_path = root / "tools/data/public_cases.json"
    weights_path = root / "tools/data/scoring_weights.json"
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
    weights = {
        str(row["anonymous_id"]): float(row["weight"])
        for row in json.loads(weights_path.read_text(encoding="utf-8"))["cases"]
    }
    perf = [c for c in cases if not c.get("correctness_only", False)]
    public_weight = sum(weights.get(str(c.get("anonymous_id", c["case_id"])), 0.0)
                        for c in perf)
    lines = [
        "# Ragged Softmax Moments - Ascend C optimisation task (Ascend 910B3)",
        "",
        "The operator consumes ragged segments of a sequence:",
        "`score[N]` (fp16), `x[N, D]` (fp16), `offsets[S+1]` (int32, contiguous,",
        "strictly increasing, `offsets[0] = 0`, `offsets[S] = N`), and `epsilon`",
        "(fp32), and produces `mean[S, D]`, `rstd[S, D]` (fp32) and",
        "`logsumexp[S]` (fp32).",
        "",
        _NUMERIC_CONTRACT,
        "",
        "## PUBLIC PERFORMANCE SHAPES (the score is the weighted geometric mean of",
        "`baseline_ms / candidate_ms` over exactly these shapes)",
        "",
        "| case | weight (renormalised) | S | N | D | length distribution |",
        "|---|---|---|---|---|---|",
    ]
    for case in perf:
        case_id = case["case_id"]
        weight = weights.get(str(case.get("anonymous_id", case_id)), 0.0)
        lines.append(
            "| {case} | {w:.4f} | {s} | {n} | {d} | {dist} |".format(
                case=case_id,
                w=(weight / public_weight) if public_weight else 0.0,
                s=case["S"], n=case["N"], d=case["D"],
                dist=case.get("length_distribution", "?"),
            )
        )
    lines += [
        "",
        "Timing is device-event based with >=500 ms conditioning, 20+ warmup",
        "launches, and per-variant medians over 4 alternating rounds, so launch",
        "overhead, host-side setup and one-off allocations are all measurable.",
        "",
        "## EDITABLE FILES",
        "",
    ]
    for path in EDITABLE_FILES:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "No new files may be added; `solution/build.sh`, `solution/run.sh`,",
        "`scripts/**`, `tools/**` and `baseline/**` are read-only for the",
        "submission and are re-supplied by the harness.",
    ]
    return "\n".join(lines) + "\n"


def write_problem_statement(template_dir: Path | str, destination: Path | str) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_problem_statement(template_dir), encoding="utf-8")
    return destination
