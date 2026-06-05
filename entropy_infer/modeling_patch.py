import transformers
from .hijack_pangu import forward_pangu_embedded_attn, dynamic_cache_update, pangu_model_forward

from torch import nn

import types

from typing import Callable, List

from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

def patch_modules(
    model: nn.Module,
    custom_forward: Callable,
    *,
    class_names: List[str] = ["Attention", "Attn"],
    match_mode: str = "fuzzy",  # "fuzzy" or "exact"
    verbose: bool = True
):
    """
    Iterate over all submodules of the model, find Attention layers, and replace their forward methods.

    Args:
        model: The loaded model instance.
        custom_forward: Your new forward function, should be compatible with the original forward signature.
                        def custom_forward(self, hidden_states, attention_mask=None, ...): ...
        class_names: List of class names to match (fuzzy match), default is ["Attention", "Attn"].
        verbose: Whether to print patch information.

    """

    patched_count = 0

    for name, module in model.named_modules():
        # 检查类名是否包含 Attention 关键词
        class_name = module.__class__.__name__
        matched = False
        if match_mode == "fuzzy":
            if any(pattern in class_name for pattern in class_names):
                matched = True
        elif match_mode == "exact":
            if class_name in class_names:
                matched = True
        else:
            raise ValueError(f"Unknown match_mode: {match_mode}. Supported modes are 'fuzzy' and 'exact'.")

        if matched:
            # 保存原始方法引用（可选，用于调试）
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward

            # 绑定新方法到实例
            module.forward = types.MethodType(custom_forward, module)
            patched_count += 1

            if verbose:
                print(f"[patch] {name} ({class_name})")

    if verbose:
        print(f"[patch] patched {patched_count} modules")

    return patched_count


def replace_pangu_attention(model):
    transformers.cache_utils.DynamicCache.update = dynamic_cache_update
    patch_modules(
        model=model,
        custom_forward=forward_pangu_embedded_attn,
        class_names=["PanguEmbeddedAttention"],
        match_mode="exact"
    )
    patch_modules(
        model=model,
        custom_forward=pangu_model_forward,
        class_names=["PanguEmbeddedModel"],
        match_mode="exact"
    )



def entropy_infer_config(model, 
                        segment_size=512, 
                        threshold_len=4096, 
                        block_size=32, 
                        prefill_budgets=2048, 
                        prefill_only=True, 
                        layer_fusion=True, 
                        layer_skip=1, 
                        decode_evict_budgets=1024, 
                        decode_fusion_mode="half", 
                        prefill_obs_score_threshold=0.9,
                        ent_based_prefill_params=None,
                        prefill_budget_type=None,
                        pooling_params=None,
                        latent_decode=False,
                        latent_decode_params=None,
                       ):
    if prefill_budget_type is None:
        if prefill_budgets > 0 and prefill_obs_score_threshold <= 0:
            prefill_budget_type = "fixed"
        elif prefill_budgets <= 0 and prefill_obs_score_threshold > 0:
            prefill_budget_type = "dynamic"
        else:
            raise ValueError(f"Invalid budget configuration: prefill_budgets={prefill_budgets}, prefill_obs_score_threshold={prefill_obs_score_threshold}. Must have exactly one of them > 0.")
        print(f"[WARNING] Prefill budget type not specified, inferring from parameters: prefill_budgets={prefill_budgets}, prefill_obs_score_threshold={prefill_obs_score_threshold}, inferred prefill_budget_type={prefill_budget_type}")
    valid_budget_types = ["dynamic", "fixed", "ent_based", "ent_binary"]
    assert prefill_budget_type in valid_budget_types, f"Invalid prefill_budget_type: {prefill_budget_type}"
    
    if prefill_budget_type == "ent_based": # check param validity
        assert ent_based_prefill_params is not None, "ent_based_prefill_params must be provided when prefill_budget_type is 'ent_based'"
        required_keys = ["e_t", "delta_t", "alpha", "base_budget"]
        for key in required_keys:
            assert key in ent_based_prefill_params, f"Missing key '{key}' in ent_based_prefill_params"
    
    for layer in model.model.layers:
        layer.self_attn.segment_size = segment_size
        layer.self_attn.threshold_len = threshold_len
        layer.self_attn.block_size = block_size
        layer.self_attn.prefill_budgets = prefill_budgets
        layer.self_attn.prefill_only = prefill_only
        layer.self_attn.layer_fusion = layer_fusion
        layer.self_attn.layer_skip = layer_skip
        layer.self_attn.decode_evict_budgets = decode_evict_budgets
        layer.self_attn.decode_fusion_mode = decode_fusion_mode
        layer.self_attn.prefill_obs_score_threshold = prefill_obs_score_threshold
        layer.self_attn.prefill_budget_type = prefill_budget_type
        layer.self_attn.ent_based_prefill_params = ent_based_prefill_params
        layer.self_attn.pooling_params = pooling_params
        layer.self_attn.latent_decode = latent_decode
        layer.self_attn.latent_decode_params = latent_decode_params
    print_params = {
        "segment_size": segment_size,
        "threshold_len": threshold_len,
        "block_size": block_size,
        "prefill_budgets": prefill_budgets,
        "prefill_only": prefill_only,
        "layer_fusion": layer_fusion,
        "layer_skip": layer_skip,
        "decode_evict_budgets": decode_evict_budgets,
        "decode_fusion_mode": decode_fusion_mode,
        "prefill_obs_score_threshold": prefill_obs_score_threshold,
        "prefill_budget_type": prefill_budget_type,
        "ent_based_prefill_params": ent_based_prefill_params,
        "pooling_params": pooling_params,
        "latent_decode": latent_decode,
        "latent_decode_params": latent_decode_params,
    }
    print(f"Entropy infer config: {print_params}", flush=True)