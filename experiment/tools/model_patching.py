def patch_models(compression_method, model=None):
    # methods
    if compression_method == "entropy_infer":
        assert model is not None, "Model instance must be provided for patching"
        from entropy_infer.modeling_patch import replace_pangu_attention
        replace_pangu_attention(model=model)
    elif compression_method is None or compression_method.lower() == "base":
        print("Base mode")
    else:
        raise ValueError(f"Unknown compression method: {compression_method}")