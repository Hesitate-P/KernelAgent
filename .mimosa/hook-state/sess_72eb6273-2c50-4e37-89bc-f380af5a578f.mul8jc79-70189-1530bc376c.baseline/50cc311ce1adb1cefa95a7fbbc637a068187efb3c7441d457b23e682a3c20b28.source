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

"""
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)

_NPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR ASCEND NPU (torch_npu + triton-ascend):**
- Always `import torch_npu` right after `import torch` (this registers the npu backend)
- Default tensor allocations to device='npu' (never 'cuda'); CPU is allowed only when necessary
- Check availability with: hasattr(torch, 'npu') and torch.npu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device; never set TRITON_BACKENDS
- Use torch.npu.synchronize() if synchronization is needed (torch_npu mirrors the torch.cuda API)
- Target hardware: Huawei Ascend 910B, HBM memory, tens of AI Cores per chip (not hundreds like CUDA SMs)
- triton-ascend manages on-chip tiling/buffering automatically; standard tl.* Triton code runs as-is"""

_NPU_KERNEL_GUIDANCE = """\
## Ascend NPU-Specific Optimizations (triton-ascend)

You are generating a Triton kernel for Huawei Ascend 910B via triton-ascend. Follow these guidelines:

1. **Device Context**: Use 'npu' as the device instead of 'cuda' (requires `import torch_npu`)
2. **Grid / Core Count**: Ascend 910B has only ~20-50 AI Cores per chip (vs hundreds of CUDA SMs).
   Prefer fewer, larger program instances; avoid over-splitting work across tiny blocks.
3. **Thread Configuration**:
   - num_warps: typically 4 or 8 (values above 8 rarely help on Ascend)
   - BLOCK_SIZE: prefer 64, 128, 256, or 512; start with 128-256 for most kernels
4. **Compute Units**: `tl.dot` maps to the Cube (matrix) unit — keep matmuls inside tl.dot so they
   hit the Cube engine; element-wise ops run on the Vector unit.
5. **Data Types**: fp16 and bf16 are the native high-throughput types; fp32 works but is slower.
   Accumulate matmul results in fp32 when precision matters.
6. **Memory**: HBM bandwidth is high but on-chip buffers (UB/L1) are small; keep working sets
   cache-friendly and prefer contiguous access patterns. Avoid excessive global-memory round trips.
7. **Synchronization**: use torch.npu.synchronize() on the host side; do not call torch.cuda.*"""

_NPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
)

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
    "npu": PlatformConfig(
        name="npu",
        device_string="npu",
        guidance_block=_NPU_GUIDANCE,
        kernel_guidance=_NPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_NPU_CUDA_HACKS,
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
