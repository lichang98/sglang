# Numeric + latency check for the fused small-M router fast path:
#   Glm4MoeGate tl.dot kernel + _biased_grouped_topk_small_m_kernel
# vs the reference chain:
#   F.linear(fp32) gate + eager biased_grouped_topk_impl semantics.
# Run on the GPU box: python3 glm_router_fastpath_check.py
import time

import torch
import torch.nn.functional as F

from sglang.srt.models.glm4_moe import _glm4_moe_gate_dot_kernel, _glm4_moe_gate_kernel
from sglang.srt.layers.moe.topk import (
    _biased_grouped_topk_small_m_kernel,
    biased_grouped_topk_gpu,
)

torch.manual_seed(0)
E, H = 160, 5120
K_ROUTED = 8
RSF = 2.5
N_EXPERTS_TOTAL = E  # fused shared expert gets id == E


def ref_router(gating, bias, G, TG, renormalize=True, apply_rsf=False):
    """Eager copy of biased_grouped_topk_impl (num_fused_shared_experts=1)."""
    scores = gating.sigmoid()
    n = scores.shape[0]
    sfc = scores + bias.unsqueeze(0)
    group_scores = sfc.view(n, G, -1).topk(2, dim=-1)[0].sum(-1)
    group_idx = torch.topk(group_scores, k=TG, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1).expand(n, G, E // G).reshape(n, -1)
    )
    tmp = sfc.masked_fill(~score_mask.bool(), float("-inf"))
    _, ids = torch.topk(tmp, k=K_ROUTED + 1, dim=-1, sorted=True)
    weights = scores.gather(1, ids)
    ids[:, -1] = E  # randint(E, E+1) is always E
    weights[:, -1] = weights[:, :-1].sum(dim=-1) / RSF
    if renormalize:
        weights = weights / weights[:, :-1].sum(dim=-1, keepdim=True)
        if apply_rsf:
            weights = weights * RSF
    return weights.float(), ids.int()


def fused_router(gating, bias, G, TG, renormalize=True, apply_rsf=False):
    M = gating.shape[0]
    EPG = E // G
    w = torch.empty((M, K_ROUTED + 1), dtype=torch.float32, device=gating.device)
    ids = torch.empty((M, K_ROUTED + 1), dtype=torch.int32, device=gating.device)
    _biased_grouped_topk_small_m_kernel[(1,)](
        gating,
        bias,
        w,
        ids,
        gating.stride(0),
        RSF,
        M,
        E=E,
        G=G,
        EPG=EPG,
        TG=TG,
        K=K_ROUTED,
        RENORM=renormalize,
        APPLY_RSF=apply_rsf,
        M_PAD=16,
        E_PAD=256,
        EPG_PAD=1 << (EPG - 1).bit_length(),
        K_PAD=8,
        num_warps=4,
    )
    return w, ids


weight_bf16 = torch.randn(E, H, dtype=torch.bfloat16, device="cuda") * 0.02
bias = torch.randn(E, dtype=torch.float32, device="cuda") * 0.1

print("=== gate dot kernel vs F.linear(fp32) ===")
for M in (1, 2, 4, 8, 16):
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(hidden.to(torch.float32), weight_bf16.to(torch.float32), None)
    out = torch.empty((M, E), dtype=torch.float32, device="cuda")
    _glm4_moe_gate_dot_kernel[(E // 32,)](
        hidden, weight_bf16, out, M, hidden.stride(0), weight_bf16.stride(0),
        N=E, K=H, M_PAD=16, BLOCK_N=32, BLOCK_K=256, num_warps=4, num_stages=4,
    )
    diff = (out - ref).abs()
    t_ref = ref.topk(8, dim=1).indices.sort(dim=1).values
    t_out = out.topk(8, dim=1).indices.sort(dim=1).values
    print(
        f"M={M:2d} max_abs={diff.max().item():.3e} "
        f"topk8_match={torch.equal(t_ref, t_out)}"
    )

print("=== fused router vs reference (ids must match exactly) ===")
fails = 0
for G, TG in ((8, 4), (8, 3), (1, 1), (16, 2)):
    for apply_rsf in (False, True):
        for M in (1, 2, 3, 4, 8, 12, 16):
            gating = torch.randn(M, E, dtype=torch.float32, device="cuda") * 3
            w_ref, id_ref = ref_router(gating, bias, G, TG, apply_rsf=apply_rsf)
            w_new, id_new = fused_router(gating, bias, G, TG, apply_rsf=apply_rsf)
            ids_ok = torch.equal(id_ref, id_new)
            w_diff = (w_ref - w_new).abs().max().item()
            ok = ids_ok and w_diff < 1e-5
            fails += not ok
            print(
                f"G={G:2d} TG={TG} apply_rsf={int(apply_rsf)} M={M:2d} "
                f"ids_match={ids_ok} max_w_diff={w_diff:.3e} {'OK' if ok else 'FAIL'}"
            )

print("=== wrapper dispatch (biased_grouped_topk_gpu) ===")
for M in (4, 16):
    gating = torch.randn(M, E, dtype=torch.float32, device="cuda") * 3
    w_ref, id_ref = ref_router(gating, bias, 8, 4)
    w_new, id_new = biased_grouped_topk_gpu(
        None,
        gating,
        bias,
        K_ROUTED + 1,
        True,
        num_expert_group=8,
        topk_group=4,
        num_fused_shared_experts=1,
        routed_scaling_factor=RSF,
        apply_routed_scaling_factor_on_output=False,
    )
    print(
        f"M={M:2d} ids_match={torch.equal(id_ref, id_new)} "
        f"max_w_diff={(w_ref - w_new).abs().max().item():.3e}"
    )

print(f"total fails: {fails}")

print("=== latency (M=4, GLM-4.6 shape G=8 TG=4) ===")
M = 4
hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
gating = torch.randn(M, E, dtype=torch.float32, device="cuda") * 3
out = torch.empty((M, E), dtype=torch.float32, device="cuda")


def bench(fn, name, iters=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    print(f"  {name}: {(time.perf_counter() - t0) / iters * 1e6:.1f} us")


bench(
    lambda: F.linear(hidden.to(torch.float32), weight_bf16.to(torch.float32), None),
    "gate F.linear(fp32)",
)
bench(
    lambda: _glm4_moe_gate_kernel[(E,)](
        hidden, weight_bf16.to(torch.float32), out, M, hidden.stride(0),
        weight_bf16.stride(0), N=E, K=H, M_PAD=16, BLOCK_K=128, num_warps=4,
    ),
    "gate triton SIMT",
)
bench(
    lambda: _glm4_moe_gate_dot_kernel[(E // 32,)](
        hidden, weight_bf16, out, M, hidden.stride(0), weight_bf16.stride(0),
        N=E, K=H, M_PAD=16, BLOCK_N=32, BLOCK_K=256, num_warps=4, num_stages=4,
    ),
    "gate triton tl.dot",
)
bench(lambda: ref_router(gating, bias, 8, 4), "router eager reference chain")
bench(lambda: fused_router(gating, bias, 8, 4), "router fused kernel")
