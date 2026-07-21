from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    per_token_group_quant_8bit,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

# Debug: SGLANG_HPC_DEBUG_MOE=1 makes the runner emulate the kernel dataflow in
# fp32 on eager calls and log per-call relative error. Requires --disable-cuda-graph
# (graph replay never reaches this Python code).
# SGLANG_HPC_MOE_USE_REF=1 goes further: RETURN the emulation instead of the kernel
# output. Same requantized weights / same fp8 activation quantization as the
# kernel dataflow, so serving quality with USE_REF isolates dataflow precision
# from kernel implementation.
_DEBUG_MOE = os.environ.get("SGLANG_HPC_DEBUG_MOE", "0") == "1"
_DEBUG_MOE_MAX = int(os.environ.get("SGLANG_HPC_DEBUG_MOE_MAX", "400"))
_DEBUG_MOE_SKIP = int(os.environ.get("SGLANG_HPC_DEBUG_MOE_SKIP", "0"))
_USE_REF_VAL = os.environ.get("SGLANG_HPC_MOE_USE_REF", "0")
_USE_REF = _USE_REF_VAL == "1"
# "triton": run sglang's real triton fused_experts (original per-channel
# weights, requires SGLANG_HPC_MOE_KEEP_ORIG=1) inside the hpc runner. A
# quality flip vs the hpc kernel isolates the hpc MoE compute itself.
_USE_REF_TRITON = _USE_REF_VAL == "triton"
# "triton_cmp": like "triton" (triton output is returned, serving stays
# coherent) but additionally computes, on the same live inputs: the hpc
# kernel on requant weights (K), the fp32 emulation on requant (E) and on
# original per-channel weights (Eo), and triton on requant-valued weights
# re-quantized per-channel (Treq). Logs per-call rel-L2 vs the triton
# output plus norm ratios. Capped by SGLANG_HPC_MOE_TRITON_CMP_MAX.
_USE_REF_TRITON_CMP = _USE_REF_VAL == "triton_cmp"
_TRITON_CMP_MAX = int(os.environ.get("SGLANG_HPC_MOE_TRITON_CMP_MAX", "300"))
# "orig": emulate with the checkpoint's original per-channel weights instead of
# the requantized blockwise ones (requires SGLANG_HPC_MOE_KEEP_ORIG=1 at load).
# Identical otherwise, so a quality flip isolates the requant as the root cause.
_REF_WEIGHTS_ORIG = os.environ.get("SGLANG_HPC_MOE_REF_WEIGHTS", "") == "orig"
_debug_moe_calls = 0
_use_ref_logged = False


def _dequant_blockwise(q, s, n, k):
    bn, bk = s.shape
    qp = torch.zeros(bn * 128, bk * 128, dtype=torch.float32, device=q.device)
    qp[:n, :k] = q.float()
    w = qp.view(bn, 128, bk, 128) * s.view(bn, 1, bk, 1)
    return w.view(bn * 128, bk * 128)[:n, :k].contiguous()


def _pc_requant_from_block(w_q, w_bs):
    """Reconstruct fp32 weights from blockwise fp8 (w_q [E,n,k], w_bs [E,bn,bk])
    and re-quantize per output channel, yielding tensors in the layout the
    triton fused_experts per-channel path expects: (q [E,n,k] fp8, s [E,n,1])."""
    E, n, k = w_q.shape
    bn, bk = (n + 127) // 128, (k + 127) // 128
    s = w_bs[:, :bn, :bk]
    qp = torch.zeros(E, bn * 128, bk * 128, dtype=torch.float32, device=w_q.device)
    qp[:, :n, :k] = w_q.float()
    w = qp.view(E, bn, 128, bk, 128) * s.view(E, bn, 1, bk, 1)
    w = w.view(E, bn * 128, bk * 128)[:, :n, :k]
    del qp
    sc = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp8_max
    q = (w / sc).clamp(fp8_min, fp8_max).to(fp8_dtype)
    return q.contiguous(), sc.contiguous()


_triton_cmp_calls = 0


def _triton_cmp(x_bf16, dispatch_output, quant_info, runner_config, fused_experts):
    """Run the diagnostic MoE matrix on live inputs; return the triton output
    on original per-channel weights (which is what gets served)."""
    global _triton_cmp_calls
    do_log = (
        not torch.cuda.is_current_stream_capturing()
        and _triton_cmp_calls < _TRITON_CMP_MAX
        and x_bf16.shape[0] > 0
        and quant_info.w13_block_scale is not None
    )
    if do_log:
        _triton_cmp_calls += 1
    t_out = fused_experts(
        hidden_states=x_bf16.clone() if do_log else x_bf16,
        w1=quant_info.w13_weight_orig,
        w2=quant_info.w2_weight_orig,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        w1_scale=quant_info.w13_scale_orig,
        w2_scale=quant_info.w2_scale_orig,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
    )
    if not do_log:
        return t_out

    n = _triton_cmp_calls
    if n == 1:
        print(
            f"[HPC-MOE-CMP] config: routed_scaling_factor="
            f"{runner_config.routed_scaling_factor} "
            f"apply_router_weight_on_input="
            f"{runner_config.apply_router_weight_on_input} "
            f"activation={runner_config.activation} "
            f"no_combine={runner_config.no_combine} "
            f"inplace={runner_config.inplace}",
            flush=True,
        )
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    x_fp8, x_scale = per_token_group_quant_8bit(
        x_bf16, group_size=128, dst_dtype=fp8_dtype
    )
    outs = {}
    outs["Eo"] = _ref_moe_emulate(
        x_fp8,
        x_scale,
        quant_info.w13_weight,
        quant_info.w13_block_scale,
        quant_info.w2_weight,
        quant_info.w2_block_scale,
        topk_ids,
        topk_weights,
        w13_orig=quant_info.w13_weight_orig,
        w13_s_orig=quant_info.w13_scale_orig,
        w2_orig=quant_info.w2_weight_orig,
        w2_s_orig=quant_info.w2_scale_orig,
    )
    outs["E"] = _ref_moe_emulate(
        x_fp8,
        x_scale,
        quant_info.w13_weight,
        quant_info.w13_block_scale,
        quant_info.w2_weight,
        quant_info.w2_block_scale,
        topk_ids,
        topk_weights,
    )
    import hpc

    outs["K"] = hpc.fuse_moe_blockwise_fp8(
        x=x_fp8,
        x_scale=x_scale,
        gate_up_weight=quant_info.w13_weight,
        gate_up_weight_scale=quant_info.w13_block_scale,
        down_weight=quant_info.w2_weight,
        down_weight_scale=quant_info.w2_block_scale,
        topk_ids=topk_ids,
        topk_scale=topk_weights,
        rank_ep=0,
        num_expert_total=quant_info.num_experts,
        shared_output=quant_info.shared_output,
    )
    w1_q, w1_s = _pc_requant_from_block(
        quant_info.w13_weight, quant_info.w13_block_scale
    )
    w2_q, w2_s = _pc_requant_from_block(
        quant_info.w2_weight, quant_info.w2_block_scale
    )
    outs["Treq"] = fused_experts(
        hidden_states=x_bf16.clone(),
        w1=w1_q,
        w2=w2_q,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
    )
    t_f = t_out.float()
    tn = t_f.norm().clamp(min=1e-12)
    msg = f"[HPC-MOE-CMP] call={n} T={x_bf16.shape[0]}"
    for key in ("Eo", "E", "K", "Treq"):
        o = outs[key].float()
        rel = ((o - t_f).norm() / tn).item()
        norm_r = (o.norm() / tn).item()
        msg += f" {key}: rel={rel:.4f} nr={norm_r:.4f}"
    print(msg, flush=True)
    return t_out


def _ref_moe_emulate(
    x_fp8,
    x_scale,
    w13_q,
    w13_bs,
    w2_q,
    w2_bs,
    topk_ids,
    topk_w,
    w13_orig=None,
    w13_s_orig=None,
    w2_orig=None,
    w2_s_orig=None,
):
    """fp32 emulation of the blockwise kernel dataflow. When w*_orig/w*_s_orig
    are given, use the original per-channel weights instead of the requantized
    blockwise ones (everything else identical)."""
    use_orig = w13_orig is not None and w2_orig is not None
    T, H = x_fp8.shape
    E = w13_q.shape[0]
    I = w2_q.shape[2]
    w13_bk = (H + 127) // 128
    w2_bk = (I + 127) // 128
    x_deq = (x_fp8.float().view(T, H // 128, 128) * x_scale.view(T, H // 128, 1)).view(
        T, H
    )
    ref = torch.zeros(T, H, dtype=torch.float32, device=x_fp8.device)
    for e in range(E):
        tok, kk = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        if use_orig:
            w13 = w13_orig[e].float() * w13_s_orig[e]  # [2I, H], scale [2I, 1]
            w2 = w2_orig[e].float() * w2_s_orig[e]  # [H, I], scale [H, 1]
        else:
            w13 = _dequant_blockwise(w13_q[e], w13_bs[e][:, :w13_bk], 2 * I, H)
            w2 = _dequant_blockwise(w2_q[e], w2_bs[e][:, :w2_bk], H, I)
        gu = (x_deq[tok] @ w13.t()).bfloat16().float()
        a = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        a_deq = torch.empty_like(a)
        for b0 in range(0, I, 128):
            blk = a[:, b0 : b0 + 128]
            s = blk.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_max
            q = (blk / (s + 1e-8)).clamp(fp8_min, fp8_max).to(fp8_dtype)
            a_deq[:, b0 : b0 + 128] = q.float() * s
        contrib = (a_deq @ w2.t()).bfloat16().float()
        ref.index_add_(0, tok, contrib * topk_w[tok, kk].unsqueeze(1).float())
    return ref


def _debug_compare(x_fp8, x_scale, w13_q, w13_bs, w2_q, w2_bs, topk_ids, topk_w, y):
    global _debug_moe_calls
    if not _DEBUG_MOE or torch.cuda.is_current_stream_capturing():
        return
    _debug_moe_calls += 1
    if (
        _debug_moe_calls <= _DEBUG_MOE_SKIP
        or _debug_moe_calls > _DEBUG_MOE_SKIP + _DEBUG_MOE_MAX
    ):
        return
    ref = _ref_moe_emulate(
        x_fp8, x_scale, w13_q, w13_bs, w2_q, w2_bs, topk_ids, topk_w
    )
    T, H = x_fp8.shape
    x_deq = (x_fp8.float().view(T, H // 128, 128) * x_scale.view(T, H // 128, 1)).view(
        T, H
    )
    diff = (y.float() - ref).abs()
    rel = (diff.norm() / ref.norm().clamp(min=1e-12)).item()
    tok_rel = (diff.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12)).max().item()
    x_absmax = x_deq.abs().max().item() if T > 0 else 0.0
    print(
        f"[HPC-MOE-DEBUG] call={_debug_moe_calls} T={T} rel={rel:.5f} "
        f"tok_rel_max={tok_rel:.5f} max={diff.max().item():.4f} "
        f"x_absmax={x_absmax:.1f} out_absmax={y.abs().max().item():.3f} "
        f"nan={bool(torch.isnan(y).any())}",
        flush=True,
    )


@dataclass
class HpcMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    num_experts: int
    shared_output: Optional[torch.Tensor] = None
    w13_block_scale: Optional[torch.Tensor] = None
    w2_block_scale: Optional[torch.Tensor] = None
    # Original per-channel weights/scales (only when SGLANG_HPC_MOE_KEEP_ORIG=1).
    w13_weight_orig: Optional[torch.Tensor] = None
    w13_scale_orig: Optional[torch.Tensor] = None
    w2_weight_orig: Optional[torch.Tensor] = None
    w2_scale_orig: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None


@register_fused_func("none", "hpc")
def fused_experts_none_to_hpc(
    dispatch_output: StandardDispatchOutput,
    quant_info: HpcMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    import hpc
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    global _use_ref_logged
    x_bf16 = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    # The standard-dispatcher contract requires the RUNNER to apply
    # routed_scaling_factor (triton applies it in moe_sum_reduce; glm4_moe
    # skips its own multiply on CUDA). The hpc kernels only apply topk
    # weights, so scale the runner output here.
    rsf = runner_config.routed_scaling_factor
    if rsf is None:
        rsf = 1.0

    if (_USE_REF_TRITON or _USE_REF_TRITON_CMP) and quant_info.w13_weight_orig is not None:
        if not _use_ref_logged:
            _use_ref_logged = True
            print(
                f"[HPC-MOE] USE_REF={_USE_REF_VAL} active: MoE = sglang triton "
                "fused_experts on original per-channel weights"
                + (" (+diagnostic matrix)" if _USE_REF_TRITON_CMP else ""),
                flush=True,
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        if _USE_REF_TRITON_CMP:
            output = _triton_cmp(
                x_bf16, dispatch_output, quant_info, runner_config, fused_experts
            )
        else:
            output = fused_experts(
                hidden_states=x_bf16,
                w1=quant_info.w13_weight_orig,
                w2=quant_info.w2_weight_orig,
                topk_output=dispatch_output.topk_output,
                moe_runner_config=runner_config,
                use_fp8_w8a8=True,
                per_channel_quant=True,
                w1_scale=quant_info.w13_scale_orig,
                w2_scale=quant_info.w2_scale_orig,
                a1_scale=quant_info.a13_scale,
                a2_scale=quant_info.a2_scale,
            )
        return StandardCombineInput(hidden_states=output)

    if quant_info.w13_block_scale is not None:
        # Blockwise path: per-token per-128-group activation scales avoid the
        # precision collapse that a single per-tensor scale causes when a batch
        # contains outlier (massive-activation) tokens; the kernel also quantizes
        # the silu(gate)*up intermediate blockwise internally.
        x_fp8, x_scale = per_token_group_quant_8bit(
            x_bf16, group_size=128, dst_dtype=fp8_dtype
        )
        if _USE_REF and not torch.cuda.is_current_stream_capturing():
            use_orig = (
                _REF_WEIGHTS_ORIG
                and quant_info.w13_weight_orig is not None
                and quant_info.w2_weight_orig is not None
            )
            if not _use_ref_logged:
                _use_ref_logged = True
                print(
                    f"[HPC-MOE] USE_REF active: MoE output = fp32 emulation, "
                    f"hpc kernel bypassed "
                    f"(weights={'orig per-channel' if use_orig else 'requantized blockwise'})",
                    flush=True,
                )
            ref = _ref_moe_emulate(
                x_fp8,
                x_scale,
                quant_info.w13_weight,
                quant_info.w13_block_scale,
                quant_info.w2_weight,
                quant_info.w2_block_scale,
                topk_ids,
                topk_weights,
                w13_orig=quant_info.w13_weight_orig if use_orig else None,
                w13_s_orig=quant_info.w13_scale_orig if use_orig else None,
                w2_orig=quant_info.w2_weight_orig if use_orig else None,
                w2_s_orig=quant_info.w2_scale_orig if use_orig else None,
            )
            output = ref.to(torch.bfloat16)
            if quant_info.shared_output is not None:
                output = output + quant_info.shared_output
            if rsf != 1.0:
                output = output * rsf
            return StandardCombineInput(hidden_states=output)

        output = hpc.fuse_moe_blockwise_fp8(
            x=x_fp8,
            x_scale=x_scale,
            gate_up_weight=quant_info.w13_weight,
            gate_up_weight_scale=quant_info.w13_block_scale,
            down_weight=quant_info.w2_weight,
            down_weight_scale=quant_info.w2_block_scale,
            topk_ids=topk_ids,
            topk_scale=topk_weights,
            rank_ep=0,
            num_expert_total=quant_info.num_experts,
            shared_output=quant_info.shared_output,
        )
        _debug_compare(
            x_fp8,
            x_scale,
            quant_info.w13_weight,
            quant_info.w13_block_scale,
            quant_info.w2_weight,
            quant_info.w2_block_scale,
            topk_ids,
            topk_weights,
            output,
        )
        if rsf != 1.0:
            output = output * rsf
        return StandardCombineInput(hidden_states=output)

    # Quantize BF16 input to FP8 with per-tensor scale.
    # HPC kernel expects per-tensor activation quantization (per-expert gate_up_scale
    # can only express a single activation scale across all tokens).
    x_scale = x_bf16.abs().max().clamp(min=1e-12) / fp8_max
    x_fp8 = (x_bf16 / x_scale).clamp(fp8_min, fp8_max).to(fp8_dtype)

    act_and_mul_scale = torch.ones(
        (1,), dtype=torch.float32, device=x_bf16.device
    )

    # Combined scales: kernel computes (fp8_x @ fp8_w.T) * scale,
    # so scale = activation_dequant_scale * weight_dequant_scale.
    gate_up_scale = x_scale * quant_info.w13_scale  # [num_experts]
    down_scale = act_and_mul_scale * quant_info.w2_scale  # [num_experts]

    output = hpc.fuse_moe_pertensor_fp8(
        x=x_fp8,
        gate_up_weight=quant_info.w13_weight,
        down_weight=quant_info.w2_weight,
        gate_up_scale=gate_up_scale,
        down_scale=down_scale,
        act_and_mul_scale=act_and_mul_scale,
        topk_ids=topk_ids,
        topk_scale=topk_weights,
        rank_ep=0,
        num_expert_total=quant_info.num_experts,
        use_bf16_mul=True,
        shared_output=quant_info.shared_output,
    )
    if rsf != 1.0:
        output = output * rsf

    return StandardCombineInput(hidden_states=output)
