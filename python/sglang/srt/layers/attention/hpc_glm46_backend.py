from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.per_token_group_quant_8bit_v2 import per_token_group_quant_8bit_v2
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.triton_ops.cache_ops import launch_reshape_and_cache_flash
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max, fp8_min
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

PAGE_SIZE = 64
HEADS_PER_GROUP = 12
# SM90 dynamic split-K launches this many CTAs per SM for num_seq_q = 1..4
# (mirrors kCtaPerSmMap in HPC-Ops sched_task_info.h).
_SM90_CTA_PER_SM = (4, 3, 3, 2)


@dataclass
class HPCGlm46DecodeMetadata:
    block_ids: torch.Tensor
    seq_lens: torch.Tensor
    task_map: torch.Tensor
    split_flag: torch.Tensor
    q_fp8: torch.Tensor
    q_scale: torch.Tensor
    output: torch.Tensor
    mtp: int = 0
    num_tokens: int = 0


@dataclass
class HPCGlm46PrefillMetadata:
    cu_seqlens_q: torch.Tensor
    block_ids: torch.Tensor
    seqlens_kvcache: torch.Tensor
    max_seqlens_q: int
    max_seqlens_q_pad: int


class HPCGlm46AttentionBackend(AttentionBackend):
    # Needs seq_lens_cpu for the adaptive split-K floor in decode metadata init.
    needs_cpu_seq_lens: bool = True

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.prefill_backend = TritonAttnBackend(model_runner)
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.page_size = model_runner.server_args.page_size
        self.ones_scale = torch.ones((1,), dtype=torch.float32, device=self.device)
        self.forward_metadata: Optional[HPCGlm46DecodeMetadata] = None
        self.prefill_metadata: Optional[HPCGlm46PrefillMetadata] = None

        hpg = self.num_q_heads // self.num_kv_heads
        if self.page_size != PAGE_SIZE:
            raise ValueError("hpc_glm46 attention requires page_size=64.")
        if self.head_dim != 128 or self.v_head_dim != 128:
            raise ValueError("hpc_glm46 attention requires qk/v head_dim=128.")
        if hpg != HEADS_PER_GROUP:
            raise ValueError("hpc_glm46 attention requires 12 Q heads per KV head.")

        self.debug_attn = os.environ.get("SGLANG_HPC_DEBUG_ATTN", "0") == "1"
        self.debug_layers = set(
            int(x)
            for x in os.environ.get("SGLANG_HPC_DEBUG_LAYERS", "0").split(",")
            if x.strip()
        )
        self.debug_max_calls = int(os.environ.get("SGLANG_HPC_DEBUG_MAX_CALLS", "50"))
        self.debug_bs = int(os.environ.get("SGLANG_HPC_DEBUG_BS", "1"))
        self.debug_max_seq = int(os.environ.get("SGLANG_HPC_DEBUG_MAX_SEQ", "8192"))
        self._debug_calls = {}

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.prefill_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        max_blocks = (self.max_context_len + PAGE_SIZE - 1) // PAGE_SIZE

        self.cuda_graph_block_ids = torch.full(
            (max_bs, max_blocks), 0, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_seq_lens = torch.empty(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_split_flag = torch.zeros(
            (max_bs, self.num_kv_heads), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_q_fp8 = torch.empty(
            (max_num_tokens, self.num_q_heads, self.head_dim),
            dtype=fp8_dtype,
            device=self.device,
        )
        self.cuda_graph_q_scale = torch.empty(
            (max_num_tokens, self.num_q_heads, 1), dtype=torch.float32, device=self.device
        )
        self.cuda_graph_output = torch.empty(
            (max_num_tokens, self.num_q_heads, self.v_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.cuda_graph_task_map = self._allocate_task_map(max_bs)
        self.cuda_graph_cu_seqlens_q = torch.empty(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._graph_buffers = True

    def _ensure_decode_buffers(self, bs: int, num_tokens: int):
        # Decode/verify metadata reuses these persistent buffers as scratch.
        # When init_cuda_graph_state ran, they are graph-owned: never realloc.
        # Otherwise (--disable-cuda-graph) allocate lazily and grow on demand —
        # note verify batches are padded (bs = padded_tokens // tokens_per_req),
        # so later calls can legitimately need larger buffers than earlier ones.
        if getattr(self, "_graph_buffers", False):
            return
        prev_bs = getattr(self, "_eager_bs", 0)
        prev_tokens = getattr(self, "_eager_tokens", 0)
        if prev_bs >= bs and prev_tokens >= num_tokens:
            return
        new_bs = max(bs, prev_bs)
        new_tokens = max(num_tokens, prev_tokens)
        max_blocks = (self.max_context_len + PAGE_SIZE - 1) // PAGE_SIZE
        self.cuda_graph_block_ids = torch.full(
            (new_bs, max_blocks), 0, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_seq_lens = torch.empty(
            (new_bs,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_split_flag = torch.zeros(
            (new_bs, self.num_kv_heads), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_q_fp8 = torch.empty(
            (new_tokens, self.num_q_heads, self.head_dim),
            dtype=fp8_dtype,
            device=self.device,
        )
        self.cuda_graph_q_scale = torch.empty(
            (new_tokens, self.num_q_heads, 1), dtype=torch.float32, device=self.device
        )
        self.cuda_graph_output = torch.empty(
            (new_tokens, self.num_q_heads, self.v_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.cuda_graph_task_map = self._allocate_task_map(new_bs)
        self._eager_bs = new_bs
        self._eager_tokens = new_tokens

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_extend()
        ):
            self.init_forward_metadata_out_graph(forward_batch)
            self.init_forward_metadata_in_graph(forward_batch)
        else:
            self.prefill_backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        is_decode_or_verify = (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        )
        if not is_decode_or_verify:
            if forward_batch.forward_mode.is_extend():
                self._init_extend_metadata(forward_batch)
                return
            self.prefill_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
            return

        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_target_verify() and forward_batch.spec_info is not None:
            num_tokens_per_req = forward_batch.spec_info.num_tokens_per_req
            if num_tokens_per_req <= 0:
                num_tokens_per_req = forward_batch.input_ids.shape[0] // bs
            mtp = num_tokens_per_req - 1
            num_tokens = bs * num_tokens_per_req
        else:
            mtp = 0
            num_tokens = bs

        self._ensure_decode_buffers(bs, num_tokens)

        seq_lens = forward_batch.seq_lens[:bs].to(torch.int32)
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens)
        if mtp > 0:
            # target_verify: seq_lens is the committed length before draft tokens;
            # the kernel (new_kv_included=True) wants total incl. all draft tokens.
            self.cuda_graph_seq_lens[:bs].add_(mtp + 1)

        block_ids = self.cuda_graph_block_ids[:bs]
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(block_ids.stride(0), PAGE_SIZE))
        ](
            self.req_to_token,
            forward_batch.req_pool_indices[:bs],
            self.cuda_graph_seq_lens[:bs],
            None,
            self.cuda_graph_block_ids,
            self.req_to_token.stride(0),
            self.cuda_graph_block_ids.stride(0),
        )

        import hpc

        # The dynamic split-K assigner floors per-CTA work at min_process_len
        # tokens (hpc default 512). At small bs*seqlen the whole KV then lands
        # on a single CTA per (request, KV head), so decode attention cost
        # grows linearly with context while most SMs idle. Drop the floor to
        # one 64-token tile when the batch cannot fill the GPU anyway.
        num_seq_q = mtp + 1
        num_ctas = self.sm_count * _SM90_CTA_PER_SM[min(num_seq_q, 4) - 1]
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = (
                int(forward_batch.seq_lens_cpu[:bs].max().item()) + num_seq_q
            )
            total_tiles = bs * ((max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE)
            min_process_len = 64 if total_tiles < 8 * num_ctas else 512
        else:
            min_process_len = 512

        hpc.assign_attention_decode_task(
            self.cuda_graph_seq_lens[:bs],
            self.cuda_graph_task_map,
            self.num_kv_heads,
            num_seq_q,
            True,
            min_process_len=min_process_len,
        )

        self.forward_metadata = HPCGlm46DecodeMetadata(
            block_ids=block_ids,
            seq_lens=self.cuda_graph_seq_lens[:bs],
            task_map=self.cuda_graph_task_map,
            split_flag=self.cuda_graph_split_flag[:bs],
            q_fp8=self.cuda_graph_q_fp8[:num_tokens],
            q_scale=self.cuda_graph_q_scale[:num_tokens],
            output=self.cuda_graph_output[:num_tokens],
            mtp=mtp,
            num_tokens=num_tokens,
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        is_decode_or_verify = (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        )
        if not is_decode_or_verify:
            if forward_batch.forward_mode.is_extend():
                return
            self.prefill_backend.init_forward_metadata_in_graph(forward_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _debug_check(
        self,
        tag: str,
        layer_id: int,
        q_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: torch.Tensor,
        seq_lens_total: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        out: torch.Tensor,
        ntpr_list,
    ):
        """Naive fp32 reference vs HPC kernel output (debug only).

        Consumes the exact inputs the kernel sees: q dequantized from the fp8
        round-trip, and KV read back from the paged cache. Prints per-token
        max/mean abs diff. Enable with SGLANG_HPC_DEBUG_ATTN=1.
        """
        if not self.debug_attn or layer_id not in self.debug_layers:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        n_calls = self._debug_calls.get(layer_id, 0)
        if n_calls >= self.debug_max_calls:
            return
        self._debug_calls[layer_id] = n_calls + 1

        try:
            num_tokens = q_fp8.shape[0]
            q_dt = (q_fp8.float() * q_scale.float()).view(
                num_tokens, self.num_kv_heads, HEADS_PER_GROUP, self.head_dim
            )
            out_v = out.float().view(
                num_tokens, self.num_kv_heads, HEADS_PER_GROUP, self.v_head_dim
            )
            ks = float(k_scale.reshape(-1)[0].item())
            vs = float(v_scale.reshape(-1)[0].item())
            scale = 1.0 / math.sqrt(self.head_dim)

            bs = min(block_ids.shape[0], len(ntpr_list), self.debug_bs)
            q_off = 0
            full_ntpr = list(ntpr_list)
            for i in range(bs):
                ntpr = int(full_ntpr[i])
                total = int(seq_lens_total[i].item())
                if total > self.debug_max_seq:
                    print(
                        f"[HPC-DEBUG] {tag} layer={layer_id} call={n_calls} req={i} "
                        f"SKIP total={total}",
                        flush=True,
                    )
                    q_off += ntpr
                    continue
                nblocks = (total + PAGE_SIZE - 1) // PAGE_SIZE
                blocks = block_ids[i, :nblocks].long()
                k_all = (
                    k_cache[blocks].view(-1, self.num_kv_heads, self.head_dim)[:total].float() * ks
                )
                v_all = (
                    v_cache[blocks].view(-1, self.num_kv_heads, self.v_head_dim)[:total].float()
                    * vs
                )
                kk = k_all.permute(1, 2, 0)  # [kv, d, total]
                vv = v_all.transpose(0, 1)  # [kv, total, vd]
                for j in range(ntpr):
                    end = total - ntpr + j + 1
                    scores = (
                        torch.einsum("ghd,gdt->ght", q_dt[q_off + j], kk[:, :, :end]) * scale
                    )
                    probs = torch.softmax(scores, dim=-1)
                    ref = torch.einsum("ght,gtv->ghv", probs, vv[:, :end, :])
                    got = out_v[q_off + j]
                    diff = (ref - got).abs()
                    print(
                        f"[HPC-DEBUG] {tag} layer={layer_id} call={n_calls} req={i} tok={j} "
                        f"total={total} max={diff.max().item():.4f} "
                        f"mean={diff.mean().item():.5f} "
                        f"ref_absmax={ref.abs().max().item():.4f} "
                        f"out_absmax={got.abs().max().item():.4f}",
                        flush=True,
                    )
                q_off += ntpr
        except Exception as e:  # never break serving from a debug hook
            print(f"[HPC-DEBUG] {tag} layer={layer_id} EXC {e!r}", flush=True)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if sinks is not None:
            raise ValueError("hpc_glm46 attention does not support attention sinks.")
        bs = forward_batch.batch_size
        if bs == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        metadata = self.forward_metadata
        assert metadata is not None
        num_tokens = metadata.num_tokens

        if save_kv_cache:
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, PAGE_SIZE, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, PAGE_SIZE, layer.tp_v_head_num, layer.v_head_dim
            )
            launch_reshape_and_cache_flash(
                k.reshape(num_tokens, layer.tp_k_head_num, layer.qk_head_dim),
                v.reshape(num_tokens, layer.tp_v_head_num, layer.v_head_dim),
                k_cache,
                v_cache,
                forward_batch.out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        q_view = q.reshape(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        per_token_group_quant_8bit_v2(
            q_view.contiguous(),
            metadata.q_fp8,
            metadata.q_scale,
            layer.qk_head_dim,
            1e-10,
            fp8_min,
            fp8_max,
        )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, PAGE_SIZE, layer.tp_k_head_num, layer.qk_head_dim
        )
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1, PAGE_SIZE, layer.tp_v_head_num, layer.v_head_dim
        )

        k_scale = layer.k_scale.reshape(1) if layer.k_scale is not None else self.ones_scale
        v_scale = layer.v_scale.reshape(1) if layer.v_scale is not None else self.ones_scale

        import hpc

        quant_type = hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR

        hpc.attention_decode_fp8(
            metadata.q_fp8,
            k_cache,
            v_cache,
            metadata.block_ids,
            metadata.seq_lens,
            metadata.q_scale.squeeze(-1),
            k_scale,
            v_scale,
            mtp=metadata.mtp,
            new_kv_included=True,
            quant_type=quant_type,
            splitk=True,
            task_map=metadata.task_map,
            split_flag=metadata.split_flag,
            output=metadata.output,
        )

        self._debug_check(
            "verify" if metadata.mtp > 0 else "decode",
            layer.layer_id,
            metadata.q_fp8,
            metadata.q_scale,
            k_cache,
            v_cache,
            metadata.block_ids,
            metadata.seq_lens,
            k_scale,
            v_scale,
            metadata.output,
            [metadata.mtp + 1] * bs,
        )

        return metadata.output.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def _init_extend_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        cu_seqlens_q = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[0] = 0
        cu_seqlens_q[1:] = torch.cumsum(
            forward_batch.extend_seq_lens[:bs].to(torch.int32), dim=0
        )

        seq_lens = forward_batch.seq_lens[:bs].to(torch.int32)

        max_seqlen_k = int(seq_lens.max().item())
        max_blocks = (max_seqlen_k + PAGE_SIZE - 1) // PAGE_SIZE
        block_ids = torch.zeros((bs, max_blocks), dtype=torch.int32, device=self.device)
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(block_ids.stride(0), PAGE_SIZE))
        ](
            self.req_to_token,
            forward_batch.req_pool_indices[:bs],
            seq_lens,
            None,
            block_ids,
            self.req_to_token.stride(0),
            block_ids.stride(0),
        )

        max_seqlens_q = int(forward_batch.extend_seq_lens[:bs].max().item())
        max_seqlens_q_pad = (max_seqlens_q + 127) // 128 * 128

        self.prefill_metadata = HPCGlm46PrefillMetadata(
            cu_seqlens_q=cu_seqlens_q,
            block_ids=block_ids,
            seqlens_kvcache=seq_lens,
            max_seqlens_q=max_seqlens_q,
            max_seqlens_q_pad=max_seqlens_q_pad,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if forward_batch.forward_mode.is_target_verify():
            return self.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )
        if forward_batch.forward_mode.is_extend():
            return self._forward_extend_hpc(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )
        return self.prefill_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
        )

    def _forward_extend_hpc(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if sinks is not None:
            raise ValueError("hpc_glm46 attention does not support attention sinks.")
        bs = forward_batch.batch_size
        num_tokens = q.shape[0]
        if bs == 0 or num_tokens == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        metadata = self.prefill_metadata
        assert metadata is not None

        if save_kv_cache:
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, PAGE_SIZE, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, PAGE_SIZE, layer.tp_v_head_num, layer.v_head_dim
            )
            launch_reshape_and_cache_flash(
                k.reshape(num_tokens, layer.tp_k_head_num, layer.qk_head_dim),
                v.reshape(num_tokens, layer.tp_v_head_num, layer.v_head_dim),
                k_cache,
                v_cache,
                forward_batch.out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        q_view = q.reshape(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        q_fp8 = torch.empty(
            num_tokens,
            layer.tp_q_head_num,
            layer.qk_head_dim,
            dtype=fp8_dtype,
            device=self.device,
        )
        q_scale = torch.empty(
            num_tokens, layer.tp_q_head_num, 1, dtype=torch.float32, device=self.device
        )
        per_token_group_quant_8bit_v2(
            q_view.contiguous(),
            q_fp8,
            q_scale,
            layer.qk_head_dim,
            1e-10,
            fp8_min,
            fp8_max,
        )

        qscale_padded = torch.zeros(
            bs,
            layer.tp_q_head_num,
            metadata.max_seqlens_q_pad,
            dtype=torch.float32,
            device=self.device,
        )
        cu_seqlens_cpu = metadata.cu_seqlens_q.cpu()
        for i in range(bs):
            start = int(cu_seqlens_cpu[i])
            end = int(cu_seqlens_cpu[i + 1])
            seq_len_i = end - start
            qscale_padded[i, :, :seq_len_i] = q_scale[start:end, :, 0].t()

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, PAGE_SIZE, layer.tp_k_head_num, layer.qk_head_dim
        )
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1, PAGE_SIZE, layer.tp_v_head_num, layer.v_head_dim
        )

        k_scale = layer.k_scale.reshape(1) if layer.k_scale is not None else self.ones_scale
        v_scale = layer.v_scale.reshape(1) if layer.v_scale is not None else self.ones_scale

        import hpc

        quant_type = hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR

        output = torch.empty(
            num_tokens,
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        hpc.attention_with_kvcache_prefill_fp8(
            q_fp8,
            k_cache,
            v_cache,
            qscale_padded,
            k_scale,
            v_scale,
            metadata.cu_seqlens_q,
            metadata.block_ids,
            metadata.seqlens_kvcache,
            metadata.max_seqlens_q,
            quant_type=quant_type,
            output=output,
        )

        self._debug_check(
            "extend",
            layer.layer_id,
            q_fp8,
            q_scale,
            k_cache,
            v_cache,
            metadata.block_ids,
            metadata.seqlens_kvcache,
            k_scale,
            v_scale,
            output,
            [int(x) for x in forward_batch.extend_seq_lens[:bs].tolist()],
        )

        return output.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def _allocate_task_map(self, max_bs: int) -> torch.Tensor:
        import hpc

        return hpc.get_attention_decode_task_workspace(
            max_bs, self.max_context_len, self.num_kv_heads
        )
