from transformers import AutoTokenizer
from datasets import load_dataset
from mini_lm import GPT, MiniLMConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")

dataset = load_dataset(
    "zrxhewenyi/50000-Mobvoi",
    cache_dir="./datasets",
    loca
)

# print(dataset)
# print(dataset["train"][10]["text"])

text = dataset["train"][10]["text"]

# print(repr(text))

ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

print(ids)
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.decode(ids))

# config = MiniLMConfig(
#     block_size=128,
#     batch_size=8,
#     n_emb=1024,
#     n_layers=16,
#     n_head=8,
# )
# gpt = GPT(config)

# num_params = sum(
#     p.numel()
#     for p in gpt.parameters()
# )

# print(f"parameters: {num_params / 1e6:.2f}M")