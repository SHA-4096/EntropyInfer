from typing import Callable, Optional, Union

import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any

try:
    import torch_npu
except Exception:
    torch_npu = None


def _is_npu_fused_attn_available() -> bool:
    if torch_npu is None:
        return False
    if not hasattr(torch, "npu"):
        return False
    try:
        if not torch.npu.is_available():
            return False
        return "910" in torch.npu.get_device_name()
    except Exception:
        return False


NPU_ATTN_INFR = _is_npu_fused_attn_available()
if NPU_ATTN_INFR:
    print("[INFO] torch_npu detected. Using NPU fused infer attention.")

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging

from transformers.masking_utils import create_causal_mask

from .ops import eattention, cache_selection
from .ops_cuda import eattention as eattention_cuda

print_info = True

def print_once(msg, flush=True):
    global print_info
    if print_info:
        print(msg, flush)
    print_info = False

from .ops import compress_kv, repeat_kv

logger = logging.get_logger(__name__)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_kv_transposed(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    The hidden states go from (batch, seqlen,
    num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads,  head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(3).expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)
    # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# copied from pangu source code, usually it's not used on npu inference
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def pangu_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    class GlobalObj:
        pass

    gobj = GlobalObj()

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        decoder_layer.self_attn._g = gobj

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    hidden_states = hidden_states[:, -1,:].unsqueeze(1)
    
    if output_hidden_states:
        assert False

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def forward_pangu_embedded_attn(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    if not hasattr(self, "perf_meta"):
        self.perf_meta = {}
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # print(f"[DEBUG] hidden_shape: {hidden_shape}, input_shape: {input_shape}", flush=True)

    query_states = self.q_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
    
    # NOTE: SHAPE: query_states, key_states, value_states: (bsz, seq_len, num_heads/num_kv_heads,head_dim)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

    q_len = input_shape[1]

    ####### Latent Decoding #######
    is_prefill = q_len != 1
    if is_prefill:
            # reset all states
            self.decoded_tokens = -1
            self.obs_query = None
            self.latent_compressed = False
    if past_key_value is not None:
        # latent_decoding_params
        latent_decode = self.latent_decode
        latent_decode_params = self.latent_decode_params
        use_gqa = latent_decode_params["use_gqa"]

        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # Update after compression, concatenate new kv with cached kv, and update cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # print(f"[DEBUG] Updating cache with new key/value states. Shape: {key_states.shape}, {value_states.shape}")
        # print(f"[DEBUG] at cache update. Q.shape: {query_states.shape}, K.shape: {key_states.shape}, V.shape: {value_states.shape}", flush=True)
        # check if critia are met
        # print(f"[DEBUG] use_gqa: {use_gqa}, latent_compressed: {getattr(self, 'latent_compressed', None)}, key_states.shape[1]: {key_states.shape[1]}, query_states.shape[1]: {query_states.shape[1]}", flush=True)
        if not use_gqa and \
                hasattr(self, "latent_compressed") and \
                self.latent_compressed and \
                key_states.shape[2] != query_states.shape[2]:
            key_states = repeat_kv_transposed(key_states, self.num_key_value_groups)
            value_states = repeat_kv_transposed(value_states, self.num_key_value_groups)
        # print(f"[DEBUG] Repeated at cache update. Q.shape: {query_states.shape}, K.shape: {key_states.shape}, V.shape: {value_states.shape}", flush=True)
        
        # NOTE: expected shape: BSND
        assert query_states.shape[2] == self.num_heads, f"Expected query_states to have num_heads in dim 2, got shape {query_states.shape}, num_heads: {self.num_heads}"
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        self.decoded_tokens += 1
        if latent_decode:
            assert latent_decode_params is not None, "latent_decode_params must be provided when latent_decode is True"
            decode_gap = latent_decode_params["decode_gap"]
            window_size = latent_decode_params["window_size"]
            assert latent_decode_params["use_gqa"] == False, "GQA is not supported in latent decode for now"
            # assert decode_gap > 0 and window_size > 0, "decode_gap and window_size must be greater than 0"
            if self.decoded_tokens <= decode_gap:
                # Observation query
                # record query for observation before compression
                if is_prefill:
                    if window_size > decode_gap:
                        keep_len = window_size - decode_gap
                    else:
                        keep_len = 1 # quick workaround
                    self.obs_query = query_states[:, -keep_len:, :, :].detach().clone() # shape: (bsz, keep_len, num_heads, head_dim)
                else:
                    # decode, append query to obs_query
                    self.obs_query = torch.cat([self.obs_query, query_states], dim=1)[:, -window_size:, :, :].detach().clone() # shape: (bsz, window_size, num_heads, head_dim)
            elif self.decoded_tokens == decode_gap + 1 or (decode_gap == -1 and not self.latent_compressed): # decode_gap = -1 corresponds to snapkv:
                if decode_gap == -1:
                    assert not hasattr(self, "obs_query") or self.obs_query is None, "obs_query should be None before snapkv decoding, got {}".format(getattr(self, "obs_query", None))
                    keep_len = window_size
                    self.obs_query = query_states[:, -keep_len:, :, :].detach().clone()
                    
                # compress kv, expect BNSD
                # print(f"[DEBUG] Before compression: Q.shape: {query_states.shape}, K.shape: {key_states.shape}, V.shape: {value_states.shape}", flush=True)
                compressed_key_states, compressed_value_states = compress_kv(self, key_states, self.obs_query, value_states)
                past_key_value.key_cache[self.layer_idx] = compressed_key_states
                past_key_value.value_cache[self.layer_idx] = compressed_value_states
                self.latent_compressed = True
                # key_states = compressed_key_states
                # value_states = compressed_value_states
                print_once(f"[DEBUG] Latent decoding triggered at token {self.decoded_tokens}. Compressed key/value shape: {compressed_key_states.shape}, {compressed_value_states.shape}", flush=True)
                # performance evaluation
                if hasattr(self.config, "efficiency_eval_type"):
                    efficiency_eval_type = self.config.efficiency_eval_type
                    # print(f"[DEBUG] Efficiency eval type: {efficiency_eval_type}", flush=True)
                    if efficiency_eval_type == "memory_profiling":
                        max_mem_before_compress = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        if not hasattr(self, "perf_meta"):
                            self.perf_meta = {}
                        self.perf_meta["max_mem_before_compress_MB"] = max_mem_before_compress
                        torch.cuda.reset_max_memory_allocated()
                    else:
                        pass
            else:
                pass
        

    ####### End of Latent Decoding #######

    attn_weights = None

    # debug
    # print(f"[DEBUG] Attention Impl: {self.config._attn_implementation}")

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    # NOTE: hard code for now
    is_prefill = q_len != 1
    do_eattn = True
    if self.prefill_only and not is_prefill:
        do_eattn = False
    criti_prefill_enabled = False
    decode_block_index = None
    block_size = None

    # NOTE: Expect query, key, value shape to be (bsz, seq_len, num_heads/num_kv_heads, head_dim)
    assert query_states.shape[1] == q_len and query_states.shape[2] == self.num_heads, f"Expected query_states shape to be (bsz, seq_len, num_heads, head_dim), got {query_states.shape}"

    # NOTE: Sparse Attention
    if self.layer_idx + 1 > self.layer_skip and do_eattn:
        # print(f"layer {self.layer_idx}, CritiPrefill, key_states shape: {key_states.shape}, value_states shape: {value_states.shape}")
        criti_prefill_enabled = True
        segment_size = self.segment_size
        threshold_len = self.threshold_len
        block_size = self.block_size
        # print("CritiPrefill")
        # if query_states.is_cuda:
        if not NPU_ATTN_INFR:
            # ops_cuda.eattention expects BSND layout on CUDA path.
            attn_output, _, decoding_block_size, decode_block_index = eattention_cuda(
                self,
                segment_size,
                threshold_len,
                block_size,
                query_states,
                key_states,
                value_states,
            )
        else:
            attn_output, _, decoding_block_size, decode_block_index = eattention(
                self, segment_size, threshold_len, block_size, query_states, key_states, value_states
            )
        assert decoding_block_size == -1 and decode_block_index == None # TODO: maybe remove it in the future

        
    else:
        # dense attention
        if not self.training and NPU_ATTN_INFR:
            q_len = input_shape[1]
            if attention_mask is not None:
                attention_mask = ~attention_mask.bool()
            elif q_len > 1:
                attention_mask = torch.triu(torch.ones([q_len, q_len]), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(query_states.device)

            actual_num_kv_heads=key_states.shape[2]

            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query_states, key_states, value_states,
                num_heads=self.num_heads, num_key_value_heads=actual_num_kv_heads,
                input_layout="BSND", atten_mask=attention_mask, scale=self.scaling)
            attn_weights = None
        else:
            local_attention_interface = attention_interface
            if query_states.is_cuda and self.config._attn_implementation != "flash_attention_2":
                print(f"[DEBUG] Using flash_attention_2 for CUDA")
                flash_attention_impl = ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
                if flash_attention_impl is not None:
                    local_attention_interface = flash_attention_impl
                else:
                    logger.warning_once(
                        "CUDA fallback requested flash_attention_2, but it is unavailable. "
                        f"Falling back to {self.config._attn_implementation}."
                    )
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            # NOTE: shape for input is (bsz, num_heads, seq_len, head_dim) for compatibility with existing attention implementations
            attn_output, attn_weights = local_attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

# transformers 4.53.2
def dynamic_cache_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        # NOTE: BSND layout — dim 1 is seq_len, dim -2 is num_heads
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[1]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                # NOTE: modified because cache dimension 1 and 2 transposed
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-3)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-3)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
