def config_compress(model, method, compression_config, golden_prompt_len = -1, args=None):
    print("====================================")
    print(f"compression method: {method}")
    print(f"compression config: {compression_config}")
    print("====================================")
    if method == "entropy_infer":
        from entropy_infer.modeling_patch import entropy_infer_config
        entropy_infer_config(model,
                    segment_size=compression_config["segment_size"],
                    threshold_len=compression_config["threshold_len"],
                    block_size=compression_config["block_size"],
                    prefill_budgets=compression_config.get("prefill_budgets", -1),
                    layer_fusion=compression_config["layer_fusion"],
                    layer_skip=compression_config["layer_skip"],
                    decode_evict_budgets=compression_config["decode_evict_budgets"],
                    prefill_obs_score_threshold=compression_config.get("prefill_obs_score_threshold", -1),
                    ent_based_prefill_params=compression_config.get("ent_based_prefill_params", None),
                    prefill_budget_type=compression_config.get("prefill_budget_type", None),
                    pooling_params=compression_config.get("pooling_params", None),
                    latent_decode=compression_config.get("latent_decode", False),
                    latent_decode_params=compression_config.get("latent_decode_params", None)
        )
    elif method == "base":
        pass
    else:
        raise ValueError(f"Unknown compression method: {method}")
    return model
