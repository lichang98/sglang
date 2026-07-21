# Numeric check for the triton Glm4MoeGate fast path vs the fp32 F.linear path.
# Run on the GPU box: python3 glm_gate_fastpath_check.py
import torch
import torch.nn.functional as F

from sglang.srt.models.glm4_moe import _glm4_moe_gate_kernel

torch.manual_seed(0)
N, K = 160, 5120
weight = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02).to(torch.float32)

for M in (1, 2, 4, 8, 16):
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    ref = F.linear(hidden.to(torch.float32), weight, None)

    out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    _glm4_moe_gate_kernel[(N,)](
        hidden, weight, out, M, hidden.stride(0), weight.stride(0),
        N=N, K=K, M_PAD=16, BLOCK_K=128, num_warps=4,
    )

    diff = (out - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    topk_ref = ref.topk(8, dim=1).indices.sort(dim=1).values
    topk_out = out.topk(8, dim=1).indices.sort(dim=1).values
    print(
        f"M={M:2d} max_abs={diff.max().item():.3e} max_rel={rel.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} topk8_match={torch.equal(topk_ref, topk_out)}"
    )

# rough latency comparison
import time

for fn, name in (
    (lambda: F.linear(hidden.to(torch.float32), weight, None), "F.linear(fp32)"),
    (
        lambda: _glm4_moe_gate_kernel[(N,)](
            hidden, weight, out, M, hidden.stride(0), weight.stride(0),
            N=N, K=K, M_PAD=16, BLOCK_K=128, num_warps=4,
        ),
        "triton fast path",
    ),
):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us  (M={M})")
