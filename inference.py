import torch
from transformers import AutoTokenizer

from mini_lm import GPT


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------
# load checkpoint
# ----------------------------

checkpoint = torch.load(
    "./checkpoints_qwen25/best_model.pt",
    map_location=device,
    weights_only=False,
)

config = checkpoint["config"]

model = GPT(config)

model.load_state_dict(
    checkpoint["model"]
)

model = model.to(device)
model.eval()

import torch
import torch.nn.functional as F


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
# ----------------------------
# tokenizer
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


# ----------------------------
# prompt
# ----------------------------

prompt = "很久很久以前"

input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
).to(device)


# ----------------------------
# generate
# ----------------------------

output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
)


text = tokenizer.decode(
    output_ids[0].tolist(),
    skip_special_tokens=True
)

print(text)