import torch

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