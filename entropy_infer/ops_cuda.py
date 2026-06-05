import math

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    repeat_kv,
)
from .tools import repeat_kv_transposed

print_info = True

def print_once(msg, flush=True):
    global print_info
    if print_info:
        print(msg, flush)
    print_info = False

def _apply_1d_pooling(scores, pooling_type, kernel_size):
    if kernel_size is None or kernel_size <= 1:
        return scores
    if pooling_type == "none":
        return scores
    if pooling_type == "maxpool":
        return F.max_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    if pooling_type == "avgpool":
        return F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    raise ValueError(f"Unknown pooling type: {pooling_type}")


def cache_profilling(fwd_obj, query_states, key_states, block_size, segment_size):
    '''
    Docstring for cache_profilling
    q,k (bs, seqlen, num_heads, head_dim)
    :param fwd_obj: Description
    :param query_states: Description
    :param key_states: Description
    :param block_size: Description
    :param segment_size: Description
    :return: prefill_index_to_keep  (bs, num_heads, num_segments, topk), decode_index_to_keep  (bs, num_heads, topk)
    '''
    fwd_obj.head_mean_profiling = True # NOTE: hardcode
    head_mean_profiling = fwd_obj.head_mean_profiling
    assert head_mean_profiling is not None, "head_mean_profiling must be provided"

    assert segment_size >= block_size

    # (bs, seqlen, num_heads, head_dim)
    bsz, seqlen, num_heads, head_dim = query_states.shape
    num_heads_k = key_states.size(-2)
    num_key_value_groups = num_heads // num_heads_k
    nrep = segment_size // block_size
    decode_evict_budgets = fwd_obj.decode_evict_budgets

    num_segments = (seqlen + segment_size - 1) // segment_size # all segments
    pool_len = (num_segments - 1) * segment_size # ignore last segment?
    num_blocks = pool_len // block_size
    num_segments_pooled = pool_len // segment_size
    num_segments_all = num_segments_pooled + 1  # including last segment
    last_query_states = query_states[:, pool_len:]
    query_states = query_states[:, :pool_len]
    key_states = key_states[:, :pool_len]


    query_states = query_states.transpose(1, 2).reshape(bsz, num_heads, pool_len//segment_size, segment_size, head_dim)
    last_query_states = last_query_states.transpose(1,2).reshape(bsz, num_heads, 1, seqlen-pool_len, head_dim)

    # NOTE: GQA support
    key_states = key_states.transpose(1, 2)
    key_states = repeat_kv(key_states, num_key_value_groups)
    key_states = key_states.reshape(bsz, num_heads, pool_len//block_size, block_size, head_dim)


    # (bs, num_heads, block_num, head_dim)
    layer_max_q = query_states.max(dim=-2).values
    layer_max_k = key_states.max(dim=-2).values
    layer_min_q = query_states.min(dim=-2).values
    layer_min_k = key_states.min(dim=-2).values

    last_max_q = last_query_states.max(dim=-2).values
    last_min_q = last_query_states.min(dim=-2).values

    layer_max_q = torch.cat([layer_max_q, last_max_q], dim=-2)
    layer_min_q = torch.cat([layer_min_q, last_min_q], dim=-2)


    # (bs, num_heads, seg_num_q, block_num_k)
    q_block_len = layer_max_q.size(-2)
    k_block_len = layer_max_k.size(-2)
    qq = torch.cat([layer_max_q, layer_min_q], dim=-2)
    attn_weights_max = torch.matmul(qq, layer_max_k.transpose(2, 3)).view(bsz, num_heads, 2, q_block_len, k_block_len).mean(dim=2)
    attn_weights_min = torch.matmul(qq, layer_min_k.transpose(2, 3)).view(bsz, num_heads, 2, q_block_len, k_block_len).mean(dim=2)
    attn_weights = torch.max(attn_weights_max, attn_weights_min)
    # Per-query-head scores: (bsz, num_heads, num_segments_all, num_blocks). Keys are already repeat_kv-expanded in profiling.
    if head_mean_profiling:
        attn_weights = attn_weights.view(bsz, num_heads_k, num_key_value_groups, num_segments_all, num_blocks).mean(dim=2)

    mask = torch.full((num_blocks+nrep, num_blocks+nrep), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)

    # NOTE: always keep the last segment, to keep first layers's accuracy and let not kv behind
    mask.masked_fill_(mask_cond + nrep < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device).view(num_segments_all, nrep, num_blocks+nrep)[:, -1, :]
    #debug: make sure truncated part is -inf
    # assert torch.all(mask[:, num_blocks:] == torch.finfo(attn_weights.dtype).min)
    mask = mask[:, :num_blocks]
    attn_weights += mask[None, None, :, :]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(key_states.dtype)

    if fwd_obj.layer_fusion == True:
        if not hasattr(fwd_obj._g, "prev_attn_weights") or fwd_obj.layer_idx <= 1:
            fwd_obj._g.prev_attn_weights = attn_weights
        else:
            pprev = 0.25
            pcurr = 1 - pprev
            cdevice = attn_weights.device
            attn_weights = fwd_obj._g.prev_attn_weights.to(cdevice) * pprev + attn_weights * pcurr
            fwd_obj._g.prev_attn_weights = attn_weights

    prefill_budget_type = fwd_obj.prefill_budget_type

    prefill_index_to_keep = None

    if prefill_budget_type == "ent_based":
        ent_based_prefill_params = fwd_obj.ent_based_prefill_params
        e_t = ent_based_prefill_params["e_t"]
        delta_t = ent_based_prefill_params["delta_t"]
        alpha = ent_based_prefill_params["alpha"]
        base_budget = ent_based_prefill_params["base_budget"]

        # Produce keep_counts based on entropy; if segment < 2, set keep_counts to base_budget
        # keep_counts = torch.zeros_like(attn_weights, dtype=torch.long)  # (bs, num_heads, num_segments)
        keep_counts = torch.zeros(attn_weights.size(0), attn_weights.size(1), attn_weights.size(2), device=attn_weights.device, dtype=torch.long)
        if num_segments_all < 2:
            keep_counts[:, :, :] = base_budget // block_size
        else:
            # Calculate entropy-based dynamic threshold
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)  # (bs, num_heads, num_segments)
            if head_mean_profiling:
                assert attn_entropy.shape == (bsz, num_heads_k, num_segments_all), f"Unexpected attn_entropy shape: {attn_entropy.shape}, expected: {(bsz, num_heads_k, num_segments_all)}"
            else:
                assert attn_entropy.shape == (bsz, num_heads, num_segments_all), f"Unexpected attn_entropy shape: {attn_entropy.shape}, expected: {(bsz, num_heads, num_segments_all)}"
            # determine budget allocation strategy for each head
            # static head: all segments' attn entropy below e_t
            assert attn_entropy.shape[-1] >= 2, f"Expected attn_entropy to have at least 2 segments, but got {attn_entropy.shape[-1]} segments only. q_len: {query_states.shape[2]}, segment_size: {segment_size}, num_segments_all: {num_segments_all}"
            # ignore first segment since it's completely masked
            static_head_mask = (attn_entropy[:, :, 1:] < e_t).all(dim=-1)  # (bs, num_heads)
            # dynamic head: some segments' attn entropy above e_t
            dynamic_head_mask = ~static_head_mask
            
            keep_budgets = torch.zeros_like(attn_entropy)

            # for static heads, kee_budget is set to base_budget
            keep_budgets[static_head_mask] = base_budget

            # for dynamic heads, calculate it based on delta_entropy
            delta_entropy_ratio = torch.zeros_like(attn_entropy)
            delta_entropy_ratio[:, :, 1:] = (attn_entropy[:, :, 1:] - attn_entropy[:, :, :-1]) / (attn_entropy[:, :, :-1] + 1e-9)
            delta_entropy_ratio[:, :, 1] = delta_t
            delta_entropy_ratio = torch.clamp(delta_entropy_ratio, 0, 1 + delta_t)
            keep_budgets[:, :, 0][dynamic_head_mask] = base_budget
            for seg_idx in range(1, num_segments_all):
                keep_budgets[:, :, seg_idx][dynamic_head_mask] = keep_budgets[:, :, seg_idx-1][dynamic_head_mask] + alpha * base_budget * (delta_entropy_ratio[:, :, seg_idx][dynamic_head_mask] - delta_t)
                keep_budgets[:, :, seg_idx][dynamic_head_mask] = torch.clamp(keep_budgets[:, :, seg_idx][dynamic_head_mask], min=base_budget, max=base_budget * 3)
            keep_counts = (keep_budgets / block_size).long()
        
        max_effective_blocks = num_blocks - (((seqlen%segment_size)//block_size) + 1)
        effective_blocks = torch.arange(num_segments_all, device=keep_counts.device) * nrep
        effective_blocks = torch.clamp(effective_blocks, max=max_effective_blocks)
        keep_counts = torch.minimum(keep_counts, effective_blocks.view(1, 1,-1))
        keep_counts[:, :, 0] = 0  # set first segment's keep_counts to 0, since it's completely masked by the causal mask

        # perform topk based on keep_counts
        segment_topk = keep_counts.amax(dim=(0, 1))  # (num_segments_all,)
        max_k = int(segment_topk.max().item())
        sorted_scores, sorted_indices = torch.sort(attn_weights, dim=-1, descending=True)
        prefill_index_to_keep = sorted_indices[..., :max_k]
        # debug; comment out in benchmark
        if not hasattr(fwd_obj, 'perf_meta'):
            fwd_obj.perf_meta = {}
        fwd_obj.segment_topk = segment_topk
    else:
        raise ValueError(f"Unknown prefill budget type: {prefill_budget_type}")

    decoding_block_size = -1
    decode_index_to_keep = None
    # for api compatibility

    return prefill_index_to_keep, decoding_block_size, decode_index_to_keep

def cache_selection(key_states, value_states, index, block_size, left_over):
    '''
    key_states.shape = (bs, seqlen, num_heads, head_dim)
    index.shape = (bs, num_heads, topk)
    '''

    bsz, seqlen, num_head, head_dim = key_states.shape

    index = index.transpose(1, 2).unsqueeze(2).unsqueeze(-1).expand(bsz, -1, block_size, num_head , head_dim)

    # debug: check index bounds
    # print(f"Max index in cache_selection: {torch.max(index)}; Allowed max: {(seqlen - left_over) // block_size - 1}", flush=True)
    # assert torch.all(index < (seqlen - left_over) // block_size), f"Index out of bounds in cache_selection, max index: {torch.max(index)}, allowed max: {(seqlen - left_over) // block_size - 1}"
    # assert left_over > 0, "left_over should be > 0"

    key_for_selection = None
    value_for_selection = None

    if left_over == 0:
        key_for_selection = key_states
        value_for_selection = value_states
    else:
        key_for_selection = key_states[:, :-left_over]
        value_for_selection = value_states[:, :-left_over]
    selected_k = torch.gather(key_for_selection.view(bsz, -1, block_size, num_head, head_dim), dim=1, index=index)
    selected_v = torch.gather(value_for_selection.view(bsz, -1, block_size, num_head , head_dim), dim=1, index=index)

    selected_k = selected_k.view(bsz, -1, num_head, head_dim)
    selected_v = selected_v.view(bsz, -1, num_head, head_dim)

    layer_selected_k = None
    layer_selected_v = None

    if left_over > 0:
        layer_selected_k = torch.cat([selected_k, key_states[:, -left_over:]], dim=1)
        layer_selected_v = torch.cat([selected_v, value_states[:, -left_over:]], dim=1)
    else:
        layer_selected_k = selected_k
        layer_selected_v = selected_v
    return layer_selected_k, layer_selected_v


def eattention(fwd_obj, segment_size, threshold_len,  block_size, query_states, key_states, value_states):
    """
    expect (bs, seq_len, num_heads, -1)
    return attn_output, index (shape: bs, num_heads, num_segments, topk)
    """
    from flash_attn import flash_attn_func
    # NOTE: q,k,v shape = (bsz, len, num_heads, head_dim)

    # index.shape = (bs, num_heads, num_segments, topk)
    prefill_block_index, decoding_block_size, decode_block_index = cache_profilling(
        fwd_obj, query_states, key_states, block_size, segment_size
    )

    input_len = query_states.size(1)
    if input_len == 1:
        assert False, "not support for now"
        # decoding
    else:
        # NOTE: full cache k, v
        attn_outputs = []
        for i in range(0, input_len, segment_size):
            q_segment = query_states[:, i:i+segment_size,:,:]
            k_segment = key_states[:, :i+segment_size,:,:]
            v_segment = value_states[:, :i+segment_size,:,:]
            # shape: (bs, seqlen, num_heads, head_dim)
            if i >= threshold_len and i + segment_size < input_len:
                # NOTE: keep last block full len
                curr_segment = i // segment_size
                # debug
                # print(f"Applying cache selection for segment starting at {i} (threshold: {threshold_len}), max_index for selction: {prefill_block_index[:, :, curr_segment, :].max()}", flush=True)
                index = prefill_block_index[:, :, curr_segment, :]
                # If prefill indices are per query head; expand GQA K/V to match for cache_selection + flash_attn.
                num_heads_q = query_states.size(2)
                num_heads_k_seg = k_segment.size(2)
                if num_heads_k_seg < num_heads_q and not fwd_obj.head_mean_profiling:
                    assert index.shape[1] == num_heads_q, f"Index shape: {index.shape}, expected q : {num_heads_q}, k : {num_heads_k_seg}"
                    n_rep = num_heads_q // num_heads_k_seg
                    # TODO: bottleneck?
                    k_segment = repeat_kv_transposed(k_segment, n_rep)
                    v_segment = repeat_kv_transposed(v_segment, n_rep)
                    # print(f"[DEBUG] using expanded index and kv cache for selection")
                if (
                    fwd_obj.prefill_budget_type == "ent_based"
                    and hasattr(fwd_obj, "segment_topk")
                ):
                    valid_k = int(fwd_obj.segment_topk[curr_segment].item())
                    if valid_k > 0:
                        index = index[:, :, :valid_k]
                        k_segment, v_segment = cache_selection(k_segment, v_segment, index, block_size, segment_size)
                else:
                    k_segment, v_segment = cache_selection(k_segment, v_segment, index, block_size, segment_size)

            attn_output = flash_attn_func(
                q_segment, k_segment, v_segment, causal=True
            )

            attn_outputs.append(attn_output)

        attn_output = torch.cat(attn_outputs, dim=1)
    return attn_output, prefill_block_index, decoding_block_size, decode_block_index


def compress_kv(self, origin_key_states, query_states, origin_value_states):
    # extract config params
    decode_evict_budgets = self.decode_evict_budgets
    pooling_method = self.latent_decode_params["pooling_method"]
    pooling_kernel_size = self.latent_decode_params["pooling_kernel_size"]
    window_size = self.latent_decode_params["window_size"]
    use_gqa = self.latent_decode_params["use_gqa"]
    gqa_func = self.latent_decode_params["gqa_func"]
    # end param extraction
    assert query_states.shape[2] >= window_size, f"Query states sequence length {query_states.shape[2]} is smaller than window size {window_size}."
    
    # print(f"[DEBUG] Shapes: Key states: {origin_key_states.shape}, Value states: {origin_value_states.shape}, Query states: {query_states.shape}", flush=True)
    # expected input shapes: (bsz, num_heads_k, seqlen, head_dim)

    key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
    value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

    bsz, num_heads, q_len, head_dim = key_states.shape

    if q_len < decode_evict_budgets:
        # print_once("no compression")
        # support gqa
        if use_gqa:
            return origin_key_states, origin_value_states
        else:
            return key_states, value_states
    else:
        attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -window_size:, -window_size:] += attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -window_size:, : -window_size].mean(dim = -2)
        
        # gqa_support 
        if use_gqa:
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
            
        if pooling_method == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = pooling_kernel_size, padding=pooling_kernel_size//2, stride=1)
        elif pooling_method == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = pooling_kernel_size, padding=pooling_kernel_size//2, stride=1)
        elif pooling_method == 'none':
            attn_cache = attn_weights_mean
        else:
            raise ValueError('Pooling method not supported')

        indices = attn_cache.topk(decode_evict_budgets - window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        # support gqa
        if use_gqa:
            k_past_compress = origin_key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            k_cur = origin_key_states[:, :, -window_size:, :]
            v_cur = origin_value_states[:, :, -window_size:, :]
        else:
            k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -window_size:, :]
            v_cur = value_states[:, :, -window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        print_once(f"(fix) compressed, head_key_states.shape={key_states.shape}, head_value_states.shape={value_states.shape}",  flush=True)
        return key_states, value_states