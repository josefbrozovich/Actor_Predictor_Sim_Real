import torch
from mamba_ssm import Mamba


class MultiHeadTransformer1BlockRoPE(torch.nn.Module):
    """
    Matches the diagram (POST-NORM):
      Self-Attn -> Residual -> LayerNorm -> MLP (position-wise) -> Residual -> LayerNorm

    x: (B, T, d)
    returns: (B, T, d)  # "logits"/features at model dim
    """
    def __init__(self, d, h, d_q, d_v, device="cpu", ffn_mult=4, causal=True):
        super().__init__()
        assert h * d_q == h * d_q  # just to emphasize shapes; d_q per head
        self.d = d
        self.h = h
        self.d_q = d_q
        self.d_v = d_v
        self.causal = causal

        # QKV projections: (B,T,d) -> (B,T,h*d_q) / (B,T,h*d_v)
        self.W_Q = torch.nn.Linear(d, h * d_q).to(device=device)
        self.W_K = torch.nn.Linear(d, h * d_q).to(device=device)
        self.W_V = torch.nn.Linear(d, h * d_v).to(device=device)

        # Attention output projection: (B,T,h*d_v) -> (B,T,d)  (so residual matches x)
        self.W_O = torch.nn.Linear(h * d_v, d).to(device=device)

        # Post-residual LayerNorms over model dim d
        self.ln1 = torch.nn.LayerNorm(d).to(device=device)
        self.ln2 = torch.nn.LayerNorm(d).to(device=device)

        # Position-wise MLP (independently per token): d -> ffn_mult*d -> d
        self.ff1 = torch.nn.Linear(d, ffn_mult).to(device=device)
        self.ff2 = torch.nn.Linear(ffn_mult, d).to(device=device)

        # output
        self.output = torch.nn.Linear(d, d_v).to(device=device)

    def apply_rope(self, x, offset=0): # 1. Added offset argument
        base = 10000 # Standard RoPE base is 10000, yours was 1000 (check if intentional)

        B, H, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # 2. Calculate actual positions using the offset
        # This ensures token at index 0 of the current chunk gets position 'offset'
        pos = torch.arange(T, device=device, dtype=dtype) + offset

        inv_freq = 1 / (base ** (torch.arange(0, D, 2, device=device, dtype=dtype) / D))

        # Outer product to get the theta matrix (T, D/2)
        theta = pos[:, None] * inv_freq[None, :]

        # Standard RoPE rotation logic
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotate_half = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

        cos_half = torch.cos(theta)
        sin_half = torch.sin(theta)

        # Reshape to (1, 1, T, D) for broadcasting
        cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2).view(1, 1, T, D)
        sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2).view(1, 1, T, D)

        return (x * cos) + (rotate_half * sin)

    def step(self, x, cache=None, pos_offset=0):
        # x: (B, T, d)
        B, T, _ = x.shape


        # 1. Project to Q, K, V
        Q = self.W_Q(x).view(B, T, self.h, self.d_q).transpose(1, 2)  # (B,h,T,d_q)
        K = self.W_K(x).view(B, T, self.h, self.d_q).transpose(1, 2)  # (B,h,T,d_q)
        V = self.W_V(x).view(B, T, self.h, self.d_v).transpose(1, 2)  # (B,h,T,d_v)

        # 2. Apply RoPE 
        # Important: We pass pos_offset so the rotary embedding 
        # matches the token's actual position in the total sequence.
        Q = self.apply_rope(Q, offset=pos_offset)
        K = self.apply_rope(K, offset=pos_offset)

        # 3. KV-Cache Logic
        if cache is not None:
            if "k" in cache and "v" in cache:
                # Append new RoPE-transformed K and raw V to history
                K = torch.cat([cache["k"], K], dim=2) 
                V = torch.cat([cache["v"], V], dim=2)
            
            cache["k"] = K
            cache["v"] = V

        # 4. Attention
        # If generating (T=1), we don't need the causal mask; 
        # the cache history is already "behind" us.
        is_causal = self.causal if T > 1 else False

        H = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, is_causal=is_causal
        )

        # 5. Output Projections & MLP
        H = H.transpose(1, 2).reshape(B, T, self.h * self.d_v)
        x = self.ln1(x + self.W_O(H))
        
        ff = torch.nn.functional.gelu(self.ff1(x))
        ff = self.ff2(ff)
        x = self.ln2(x + ff)

        logits = self.output(x)

        return logits, x, cache

    def forward(self, x, cache=None, pos_offset=0):
        # x: (B, T, d)
        B, T, _ = x.shape


        # 1. Project to Q, K, V
        Q = self.W_Q(x).view(B, T, self.h, self.d_q).transpose(1, 2)  # (B,h,T,d_q)
        K = self.W_K(x).view(B, T, self.h, self.d_q).transpose(1, 2)  # (B,h,T,d_q)
        V = self.W_V(x).view(B, T, self.h, self.d_v).transpose(1, 2)  # (B,h,T,d_v)

        # 2. Apply RoPE 
        # Important: We pass pos_offset so the rotary embedding 
        # matches the token's actual position in the total sequence.
        Q = self.apply_rope(Q, offset=pos_offset)
        K = self.apply_rope(K, offset=pos_offset)

        # 3. KV-Cache Logic
        if cache is not None:
            if "k" in cache and "v" in cache:
                # Append new RoPE-transformed K and raw V to history
                K = torch.cat([cache["k"], K], dim=2) 
                V = torch.cat([cache["v"], V], dim=2)
            
            cache["k"] = K
            cache["v"] = V

        # 4. Attention
        # If generating (T=1), we don't need the causal mask; 
        # the cache history is already "behind" us.
        is_causal = self.causal if T > 1 else False

        H = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, is_causal=is_causal
        )

        # 5. Output Projections & MLP
        H = H.transpose(1, 2).reshape(B, T, self.h * self.d_v)
        x = self.ln1(x + self.W_O(H))
        
        ff = torch.nn.functional.gelu(self.ff1(x))
        ff = self.ff2(ff)
        x = self.ln2(x + ff)

        logits = self.output(x)

        return logits, x


class MambaPolicy(torch.nn.Module):
    def __init__(self, d, d_state, d_conv, expand, d_v, device="cpu"):
        super().__init__()

        self.d = d
        self.d_v = d_v
        self.device = torch.device(device)

        # Mamba block
        self.mamba = Mamba(
            d_model=d,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        ).to(self.device)

        self.mamba.layer_idx = 0

        self.norm = torch.nn.LayerNorm(d).to(self.device)

        # Final projection to action logits
        self.W_out = torch.nn.Linear(d, d_v).to(self.device)

        self.layer_idx = 0

    def forward(self, state):
        """
        state: (B, T, d)
        returns: probs (B, T, d_v)
        """
        # Mamba is causal by construction
        x = self.mamba(state)          # (B, T, d)
        x = self.norm(x)

        logits = self.W_out(x)         # (B, T, d_v)

        return logits, x

    def allocate_cache(self, batch_size, max_seqlen, dtype):
        return self.mamba.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            device=self.device,
            dtype=dtype
        )
    
    def reset_cache_(self, cache):
        if isinstance(cache, dict):
            for v in cache.values():
                if torch.is_tensor(v):
                    v.zero_()
                elif isinstance(v, (list, tuple)):
                    for t in v:
                        if torch.is_tensor(t):
                            t.zero_()
        else:
            for name in dir(cache):
                if name.startswith("_"):
                    continue
                try:
                    v = getattr(cache, name)
                except Exception:
                    continue
                if torch.is_tensor(v):
                    v.zero_()


    @torch.no_grad()
    def step(self, x_t, inference_params=None):
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        y = self.mamba(x_t, inference_params=inference_params)
        y = self.norm(y)
        # getting the current hidden value
        # Batch, d_model*expand, N
        _, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]

        return self.W_out(y[:, -1, :]), ssm_state



class MLP_4(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, device="cpu"):
        """
        input dim shouls be observation space
        output dim is action space
        hidden_dim_1 
        hidden_dim_2
        hidden_dim_3
        """
        super().__init__()
        self.device = torch.device(device)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim_1, device=device)
        self.linear2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2, device=device)
        self.linear3 = torch.nn.Linear(hidden_dim_2, hidden_dim_3, device=device)
        self.linear4 = torch.nn.Linear(hidden_dim_3, output_dim, device=device)

    def forward(self, x):
        h_1 = torch.relu(self.linear1(x))
        h_2 = torch.relu(self.linear2(h_1))
        h_3 = torch.relu(self.linear3(h_2))
        # probs = torch.nn.functional.softmax(self.linear4(h_3), dim=-1)

        logits = self.linear4(h_3)

        return logits, None


class MLP_3(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1, hidden_dim_2, device="cpu"):
        """
        input dim shouls be observation space
        output dim is action space
        hidden_dim_1 
        hidden_dim_2
        hidden_dim_3
        """
        super().__init__()
        self.device = torch.device(device)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim_1, device=device)
        self.linear2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2, device=device)
        self.linear3 = torch.nn.Linear(hidden_dim_2, output_dim, device=device)
#        self.linear4 = torch.nn.Linear(hidden_dim_3, output_dim, device=device)

    def forward(self, x):
        h_1 = torch.relu(self.linear1(x))
        h_2 = torch.relu(self.linear2(h_1))
        logits = self.linear3(h_2)
        # probs = torch.nn.functional.softmax(self.linear4(h_3), dim=-1)
#        logits = self.linear4(h_3)

        return logits, None
    
    def step(self, x):
        h_1 = torch.relu(self.linear1(x))
        h_2 = torch.relu(self.linear2(h_1))
        logits = self.linear3(h_2)
        # probs = torch.nn.functional.softmax(self.linear4(h_3), dim=-1)
#        logits = self.linear4(h_3)

        return logits, None



class FF(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1, hidden_dim_2, device="cpu"):
        """
        Args:
            input_dim (int): state dimension
            output_dim (int): number of actions
            hidden_dim_1 (int): dimensiont of h_1
            hidden_dim_2 (int): dimension of h_2
        """
        super().__init__()
        self.device = torch.device(device)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim_1).to(self.device)
        self.linear2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2).to(self.device)
        self.linear3 = torch.nn.Linear(hidden_dim_2, output_dim).to(self.device)



    def forward(self, state):
        """
        Args:
            state (torch.Tensor): state, 2-D tensor of shap (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)            
        """
        h_1 = torch.nn.functional.relu(self.linear1(state))
        h_2 = torch.nn.functional.relu(self.linear2(h_1))

        # probs = torch.nn.functional.softmax(self.linear3(h_2)/1000, dim=1)

        probs = torch.nn.functional.softmax(self.linear3(h_2)/2000, dim=1)


        entropy = -torch.sum(probs*torch.log(probs+1e-9), dim=1)

        action = torch.multinomial(probs, num_samples=1).squeeze()

        return action, entropy

    def forward_2(self, state):
        """
        Args:
            state (torch.Tensor): state, 2-D tensor of shap (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)            
        """
        h_1 = torch.nn.functional.relu(self.linear1(state))
        h_2 = torch.nn.functional.relu(self.linear2(h_1))

        # probs = torch.nn.functional.softmax(self.linear3(h_2)/1000, dim=1)

        logits = self.linear3(h_2)

        return logits





