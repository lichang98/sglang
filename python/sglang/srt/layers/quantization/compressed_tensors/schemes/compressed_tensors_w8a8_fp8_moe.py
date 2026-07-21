from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_moe_weight_sizes,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    is_fp8_fnuz,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    per_tensor_dequantize,
    swap_w13_to_w31,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A8Fp8MoE"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight


logger = logging.getLogger(__name__)


def _requant_fp8_weight_blockwise(
    weight_fp8: torch.Tensor,
    weight_scale_per_channel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Dequantize per-channel fp8 weights, requantize per 128x128 block.
    # Handles dims not divisible by 128 (partial last blocks) — the GEMM
    # kernel's TMA bounds-checking zero-fills OOB elements in the last tile,
    # so weights are stored unpadded. Zero padding only affects scale
    # computation (amax of zeros = 0, no effect on real-block scales).
    # weight_fp8: [n, k] fp8, weight_scale_per_channel: [n, 1] float32.
    # Returns (fp8 weight [n, k] unpadded, block scales [ceil(n/128), ceil(k/128)]).
    w = per_tensor_dequantize(weight_fp8, weight_scale_per_channel)
    n, k = w.shape
    bn, bk = (n + 127) // 128, (k + 127) // 128
    pad_n, pad_k = bn * 128, bk * 128
    if (n, k) != (pad_n, pad_k):
        padded = torch.zeros(pad_n, pad_k, dtype=w.dtype, device=w.device)
        padded[:n, :k] = w
        w = padded
    wb = w.view(bn, 128, bk, 128)
    amax = wb.abs().to(torch.float32).amax(dim=(1, 3)).clamp_(min=1e-12)
    scales = amax / fp8_max
    q = (wb / scales.view(bn, 1, bk, 1)).clamp_(fp8_min, fp8_max).to(fp8_dtype)
    q = q.view(pad_n, pad_k)[:n, :k].contiguous()
    return q, scales


class CompressedTensorsW8A8Fp8MoE(CompressedTensorsMoEScheme):

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()

        per_tensor = (
            self.weight_quant.strategy == QuantizationStrategy.TENSOR
            and self.input_quant.strategy == QuantizationStrategy.TENSOR
        )
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not (per_tensor or per_channel):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_parallel().tp_size
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=False,
        )

        extra_weight_attrs.update(
            {"weight_padded": weight_padded},
        )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_up_dim,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_quant.strategy}"
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            assert (
                self.input_quant.strategy == QuantizationStrategy.TENSOR
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module | FusedMoE) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        # Requantize per-channel weights for the HPC backend.
        if (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and get_moe_runner_backend().is_hpc()
        ):
            n13, k13 = layer.w13_weight.shape[1], layer.w13_weight.shape[2]
            n2, k2 = layer.w2_weight.shape[1], layer.w2_weight.shape[2]
            inter = k2
            # TP-sharded intermediate (e.g. 1536/8=192) may not be a 128 multiple.
            # The HPC blockwise GEMM kernel's TMA bounds-checking zero-fills OOB
            # elements in the last partial 128-wide k-tile, so weights are stored
            # unpadded. Each partial block gets its own scale (amax over real
            # elements only; zero-padding contributes 0 to amax).
            use_blockwise = k13 % 128 == 0 and n2 % 128 == 0 and n13 == 2 * inter
            if use_blockwise:
                # 128x128-block weight scales + per-token-block activation scales.
                # The per-tensor variant collapses precision on long batches with
                # massive-activation outlier tokens (one global activation scale).
                def _ceil4(x: int) -> int:
                    return (x + 3) // 4 * 4

                w13_bn = (n13 + 127) // 128
                w13_bk = (k13 + 127) // 128
                w2_bn = (n2 + 127) // 128
                w2_bk = (k2 + 127) // 128
                w13_q = torch.empty(
                    layer.num_local_experts,
                    n13,
                    k13,
                    dtype=layer.w13_weight.dtype,
                    device=layer.w13_weight.device,
                )
                w2_q = torch.empty(
                    layer.num_local_experts,
                    n2,
                    k2,
                    dtype=layer.w2_weight.dtype,
                    device=layer.w2_weight.device,
                )
                w13_bs = torch.empty(
                    layer.num_local_experts,
                    w13_bn,
                    _ceil4(w13_bk),
                    dtype=torch.float32,
                    device=layer.w13_weight.device,
                )
                w2_bs = torch.empty(
                    layer.num_local_experts,
                    w2_bn,
                    _ceil4(w2_bk),
                    dtype=torch.float32,
                    device=layer.w2_weight.device,
                )
                for expert_id in range(layer.num_local_experts):
                    # Requant the entire gate_up [n13, k13] as one piece. For
                    # GLM-4.6 (n13=384=3*128) this yields 3 n-blocks aligned to
                    # 128-tiling, vs 4 misaligned n-blocks (2+2) if gate/up
                    # halves were requanted separately.
                    q13, s13 = _requant_fp8_weight_blockwise(
                        layer.w13_weight[expert_id],
                        layer.w13_weight_scale[expert_id],
                    )
                    w13_q[expert_id] = q13
                    w13_bs[expert_id, :, : s13.shape[1]] = s13
                    if s13.shape[1] < w13_bs.shape[2]:
                        w13_bs[expert_id, :, s13.shape[1] :] = 1.0
                    q2, s2 = _requant_fp8_weight_blockwise(
                        layer.w2_weight[expert_id],
                        layer.w2_weight_scale[expert_id],
                    )
                    w2_q[expert_id] = q2
                    w2_bs[expert_id, :, : s2.shape[1]] = s2
                    if s2.shape[1] < w2_bs.shape[2]:
                        w2_bs[expert_id, :, s2.shape[1] :] = 1.0
                if get_bool_env_var("SGLANG_HPC_MOE_KEEP_ORIG"):
                    # Stash the original per-channel weights/scales so the
                    # runner's fp32 emulation can isolate the requant's
                    # precision impact (SGLANG_HPC_MOE_REF_WEIGHTS=orig).
                    layer.w13_weight_orig = layer.w13_weight
                    layer.w13_weight_scale_orig = layer.w13_weight_scale
                    layer.w2_weight_orig = layer.w2_weight
                    layer.w2_weight_scale_orig = layer.w2_weight_scale
                layer.w13_weight = torch.nn.Parameter(w13_q, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_bs.amax(dim=[1, 2]), requires_grad=False
                )
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_bs.amax(dim=[1, 2]), requires_grad=False
                )
                layer.w13_block_scale = w13_bs
                layer.w2_block_scale = w2_bs
            else:
                # HPC kernel only supports per-expert (per-tensor) dequant scales.
                max_w13_scales = layer.w13_weight_scale.amax(dim=[1, 2])  # [E]
                max_w2_scales = layer.w2_weight_scale.amax(dim=[1, 2])  # [E]
                for expert_id in range(layer.num_local_experts):
                    dq_w13 = per_tensor_dequantize(
                        layer.w13_weight[expert_id],
                        layer.w13_weight_scale[expert_id],
                    )
                    (
                        layer.w13_weight[expert_id],
                        _,
                    ) = scaled_fp8_quant(dq_w13, max_w13_scales[expert_id])
                    dq_w2 = per_tensor_dequantize(
                        layer.w2_weight[expert_id],
                        layer.w2_weight_scale[expert_id],
                    )
                    (
                        layer.w2_weight[expert_id],
                        _,
                    ) = scaled_fp8_quant(dq_w2, max_w2_scales[expert_id])
                layer.w13_weight_scale = torch.nn.Parameter(
                    max_w13_scales, requires_grad=False
                )
                layer.w2_weight_scale = torch.nn.Parameter(
                    max_w2_scales, requires_grad=False
                )

        if self.weight_quant.strategy == QuantizationStrategy.CHANNEL and _use_aiter:
            with torch.no_grad():
                # Pre-shuffle weights
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()

        if (
            self.weight_quant.strategy == QuantizationStrategy.BLOCK
            and self.use_flashinfer_trtllm
        ):
            layer.w13_weight = torch.nn.Parameter(
                swap_w13_to_w31(layer.w13_weight.data),
                requires_grad=False,
            )
            layer.w13_weight_scale = torch.nn.Parameter(
                swap_w13_to_w31(layer.w13_weight_scale.data),
                requires_grad=False,
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto():
            if (
                _use_aiter
                and self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                and get_moe_a2a_backend().supports_aiter()
            ):
                moe_runner_backend = MoeRunnerBackend.AITER
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON

        if (
            moe_runner_backend.is_aiter()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
            or moe_runner_backend.is_hpc()
        ):
            if moe_runner_backend.is_hpc():
                import sglang.srt.layers.moe.moe_runner.hpc  # noqa: F401
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if self.runner.runner_backend.is_hpc():
            from sglang.srt.layers.moe.moe_runner.hpc import HpcMoeQuantInfo

            if self.weight_quant.strategy == QuantizationStrategy.BLOCK:
                w13_scale = 1.0 / layer.w13_weight_scale.amin(dim=[1, 2])
                w2_scale = 1.0 / layer.w2_weight_scale.amin(dim=[1, 2])
            elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
                if layer.w13_weight_scale.dim() > 1:
                    w13_scale = layer.w13_weight_scale.amax(dim=[1, 2])
                    w2_scale = layer.w2_weight_scale.amax(dim=[1, 2])
                else:
                    w13_scale = layer.w13_weight_scale
                    w2_scale = layer.w2_weight_scale
            else:
                w13_scale = layer.w13_weight_scale
                w2_scale = layer.w2_weight_scale

            quant_info = HpcMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                num_experts=layer.num_experts,
                w13_block_scale=getattr(layer, "w13_block_scale", None),
                w2_block_scale=getattr(layer, "w2_block_scale", None),
                w13_weight_orig=getattr(layer, "w13_weight_orig", None),
                w13_scale_orig=getattr(layer, "w13_weight_scale_orig", None),
                w2_weight_orig=getattr(layer, "w2_weight_orig", None),
                w2_scale_orig=getattr(layer, "w2_weight_scale_orig", None),
                a13_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
            )
            return self.runner.run(dispatch_output, quant_info)

        if self.runner.runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                AiterMoeQuantInfo,
                AiterQuantType,
            )

            assert not moe_runner_config.no_combine, "unsupported"
            quant_info = AiterMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                quant_type=AiterQuantType.PER_TOKEN,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            if self.use_flashinfer_trtllm:
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    get_activation_type,
                )

                activation_type = get_activation_type(
                    moe_runner_config.activation,
                    is_gated=moe_runner_config.is_gated,
                )
                quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    global_num_experts=layer.num_experts,
                    local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                    local_num_experts=layer.num_local_experts,
                    intermediate_size=layer.w2_weight.shape[2],
                    routing_method_type=layer.routing_method_type,
                    block_quant=self.block_quant,
                    weight_block_k=self.weight_block_size[1],
                    w13_weight_scale_inv=layer.w13_weight_scale,
                    w2_weight_scale_inv=layer.w2_weight_scale,
                    activation_type=activation_type,
                )
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    use_fp8_w8a8=True,
                    w13_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a13_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                    block_shape=self.weight_block_size,
                )
            return self.runner.run(dispatch_output, quant_info)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_quant.strategy
                == QuantizationStrategy.CHANNEL,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)
