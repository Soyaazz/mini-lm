import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch

# print(torch.__version__)
@dataclass
class MiniLMConfig:
    block_size : int = 512
    batch_size : int = 32
    n_emb : int = 512
    n_layers : int = 12
    n_head : int = 8
    drop_out : float = 0.1
    #GPT-2 vocab size
    vocab_size: int = 50257
    
    @property
    def head_size(self):
        return self.n_emb // self.n_head
    
    @property
    def intermediate_size(self):
        return 4 * self.n_emb

# MHA
# FFN
# Transformer Block
# Full GPT Model
class HultiHeadAttention(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.n_emb, config.n_emb)
        self.k_proj = nn.Linear(config.n_emb, config.n_emb)
        self.v_proj = nn.Linear(config.n_emb, config.n_emb)
        self.o_proj = nn.Linear(config.n_emb, config.n_emb)
        
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.drop_out)
    
    def forward(self, x):
        B, S, hidden_states = x.shape
        d = hidden_states // self.config.n_head
        
        q = self.q_proj(x).view(B, S, self.config.n_head, self.config.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.config.n_head, self.config.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.config.n_head, self.config.head_size).transpose(1, 2)
        
        weight = (q @ k.transpose(-2, -1)) * (1.0 / (d ** 0.5))
        weight = weight.masked_fill(self.attention_mask[:S, :S] == 0, float('-inf'))
        weight = torch.softmax(weight, dim=-1)
        attention = self.dropout(weight) @ v
        attention = attention.transpose(1, 2).contiguous().view(B, S, self.config.n_emb)
        output = self.o_proj(attention)
        return output
    
class FFN(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_emb, config.intermediate_size)
        self.up_proj   = nn.Linear(config.n_emb, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_emb)
        

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.pre_attn_layer_norm = nn.RMSNorm(config.n_emb, eps=1e-6)
        self.pre_ffn_layer_norm = nn.RMSNorm(config.n_emb, eps=1e-6)
        
        self.attn = HultiHeadAttention(config)
        self.ffn = FFN(config)
        
    def forward(self, x):
        residual = x
        x = self.pre_attn_layer_norm(x)
        x = self.attn(x) + x
        x = self.pre_ffn_layer_norm(x)
        x = x + self.ffn(x)
        return x
    
class GPT(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_emb)
        self.position_embedding = nn.Embedding(config.block_size, config.n_emb)
        
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.RMSNorm(config.n_emb, eps=1e-6)
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets):
        x = self.token_embedding(x)
        pos_embedding = self.position_embedding(
            torch.arange(x.size(1), device=x.device)
        )
        x = x + pos_embedding
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            # logits的shape是(B, S, vocab_size)
            B, S, _ = logits.shape
            loss = F.cross_entropy(
                logits.view(B * S, -1),
                targets.view(B * S)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        model,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
    ):
        model.eval()

        for _ in range(max_new_tokens):

            # 如果长度超过模型最大 context，只保留最后 block_size 个 token
            idx_cond = input_ids[:, -model.config.block_size:]

            # 你的 forward 返回 logits, loss
            logits, _ = model(idx_cond)

            # 只取最后一个 token 的预测
            # [B, S, vocab] -> [B, vocab]
            logits = logits[:, -1, :]

            # temperature
            logits = logits / temperature

         # top-k sampling
            if top_k is not None:
                k = min(top_k, logits.size(-1))

                values, _ = torch.topk(logits, k)

                threshold = values[:, [-1]]

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            # logits -> probability
            probs = F.softmax(logits, dim=-1)

        # 按概率采样下一个 token
            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

        # [B, T] + [B, 1]
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        # 遇到 EOS 停止
            if (
                eos_token_id is not None
                and torch.all(next_token == eos_token_id)
            ):
                break

        return input_ids
    
    def emb_tokens(self, x):
        return self.token_embedding(x)