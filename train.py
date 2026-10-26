import os
import json
import math
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from mini_lm import GPT, MiniLMConfig


# ============================================================
# 0. User config
# ============================================================

# Chinese-friendly tokenizer. IMPORTANT: changing tokenizer means training from scratch.
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

# Mixed Chinese corpus: TinyStories-Zh + Chinese Wikipedia + Zhihu.
DATASET_NAME = "richarddzh/chinese-small-lm-corpus"
DATASET_SPLIT = "train"

# Use a NEW directory so old GPT-2 token files are never reused accidentally.
TOKEN_DIR = "./processed_qwen25_chinese"
TRAIN_BIN = os.path.join(TOKEN_DIR, "train_tokens.bin")
VAL_BIN = os.path.join(TOKEN_DIR, "val_tokens.bin")
META_JSON = os.path.join(TOKEN_DIR, "meta.json")

CHECKPOINT_DIR = "./checkpoints_qwen25"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pt")
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# Model. Because Qwen tokenizer has a large vocabulary, 768 hidden keeps the
# embedding matrix from making the model unnecessarily huge.
BLOCK_SIZE = 512
N_EMB = 768
N_LAYERS = 12
N_HEAD = 12
DROP_OUT = 0.1
TIE_WORD_EMBEDDINGS = True

# Training memory settings. If CUDA OOM: reduce MICRO_BATCH_SIZE first.
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 2

# Optimizer / schedule (optimizer-step based, NOT micro-batch based).
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2_000
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

# Data preprocessing.
VAL_DOCS = 10_000
TOKENIZE_BATCH_SIZE = 64
SHUFFLE_BUFFER = 20_000

# Loader / logging.
NUM_WORKERS = 4
LOG_EVERY = 100          # optimizer steps
EVAL_EVERY = 5_000       # optimizer steps
SAVE_EVERY = 5_000       # optimizer steps; always overwrites latest.pt
VAL_MAX_BATCHES = 300

SEED = 42


# ============================================================
# 1. Reproducibility / device
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# 2. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
)

# We only use the tokenizer. Context length is controlled by our MiniLM.
tokenizer.model_max_length = 10**9

if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer has no eos_token_id; please choose a document separator token explicitly.")

VOCAB_SIZE = len(tokenizer)
EOS_ID = tokenizer.eos_token_id

print("Tokenizer:", TOKENIZER_NAME)
print("Vocab size:", VOCAB_SIZE)
print("EOS token id:", EOS_ID)

# Qwen token ids exceed uint16 range, so use uint32 on disk.
TOKEN_DTYPE = np.uint32
TOKEN_DTYPE_NAME = "uint32"

if VOCAB_SIZE > np.iinfo(TOKEN_DTYPE).max:
    raise RuntimeError(f"vocab_size={VOCAB_SIZE} does not fit in {TOKEN_DTYPE_NAME}")


# ============================================================
# 3. One-time corpus preprocessing -> uint32 .bin files
# ============================================================

def write_token_stream(dataset_iter, output_path, batch_size=64):
    token_count = 0
    doc_count = 0
    texts = []

    def process_batch(batch_texts, file_obj):
        nonlocal token_count
        if not batch_texts:
            return

        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]

        for ids in encoded:
            # Separate documents explicitly.
            ids.append(EOS_ID)
            arr = np.asarray(ids, dtype=TOKEN_DTYPE)
            arr.tofile(file_obj)
            token_count += len(ids)

    with open(output_path, "wb") as f:
        for sample in dataset_iter:
            text = sample.get("text")
            if text is None:
                continue
            text = text.strip()
            if not text:
                continue

            texts.append(text)
            doc_count += 1

            if len(texts) >= batch_size:
                process_batch(texts, f)
                texts.clear()

                if doc_count % 10_000 == 0:
                    print(
                        f"  processed docs={doc_count:,}, "
                        f"tokens={token_count / 1e6:.2f}M"
                    )

        process_batch(texts, f)

    print(
        f"Finished {output_path}: "
        f"docs={doc_count:,}, tokens={token_count / 1e6:.2f}M"
    )
    return token_count, doc_count


def token_files_are_compatible():
    if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN) and os.path.exists(META_JSON)):
        return False

    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected = {
        "tokenizer_name": TOKENIZER_NAME,
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_ID,
        "dtype": TOKEN_DTYPE_NAME,
    }

    for k, v in expected.items():
        if meta.get(k) != v:
            raise RuntimeError(
                f"Existing token files are incompatible: meta[{k}]={meta.get(k)!r}, "
                f"expected {v!r}. Delete {TOKEN_DIR} and preprocess again."
            )

    return True


def prepare_dataset():
    os.makedirs(TOKEN_DIR, exist_ok=True)

    if token_files_are_compatible():
        print("Compatible token files already exist; skipping tokenization.")
        return

    print("Loading Chinese dataset in streaming mode...")
    ds = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # Deterministic approximate shuffle for iterable dataset.
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    val_stream = ds.take(VAL_DOCS)
    train_stream = ds.skip(VAL_DOCS)

    print("Building validation token file...")
    val_tokens, val_docs = write_token_stream(
        val_stream,
        VAL_BIN,
        TOKENIZE_BATCH_SIZE,
    )

    print("Building training token file...")
    train_tokens, train_docs = write_token_stream(
        train_stream,
        TRAIN_BIN,
        TOKENIZE_BATCH_SIZE,
    )

    meta = {
        "tokenizer_name": TOKENIZER_NAME,
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_ID,
        "dtype": TOKEN_DTYPE_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": train_docs,
        "val_docs": val_docs,
    }

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


prepare_dataset()


# ============================================================
# 4. Memory-mapped causal-LM dataset
# ============================================================

class TokenBinDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(
            path,
            dtype=TOKEN_DTYPE,
            mode="r",
        )
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start:start + self.block_size + 1]

        # copy=True avoids PyTorch warning about non-writable memmap views.
        chunk = np.array(chunk, dtype=np.int64, copy=True)
        chunk = torch.from_numpy(chunk)

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


train_dataset = TokenBinDataset(TRAIN_BIN, BLOCK_SIZE)
val_dataset = TokenBinDataset(VAL_BIN, BLOCK_SIZE)

print(f"Train tokens: {len(train_dataset.data):,}")
print(f"Val tokens:   {len(val_dataset.data):,}")
print(f"Train blocks: {len(train_dataset):,}")

pin_memory = device.type == "cuda"


def make_train_loader(epoch: int):
    # Recreating the same epoch loader with the same seed reconstructs the same
    # shuffle order. This lets us resume from batch_in_epoch deterministically.
    g = torch.Generator()
    g.manual_seed(SEED + epoch)

    return DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=g,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


val_loader = DataLoader(
    val_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
)

# Number of optimizer updates in a complete epoch. We intentionally ignore an
# incomplete final accumulation group.
_probe_loader = make_train_loader(0)
MICRO_BATCHES_PER_EPOCH = len(_probe_loader)
USABLE_MICRO_BATCHES = (
    MICRO_BATCHES_PER_EPOCH // GRAD_ACCUM_STEPS
) * GRAD_ACCUM_STEPS
OPT_STEPS_PER_EPOCH = USABLE_MICRO_BATCHES // GRAD_ACCUM_STEPS
PLANNED_TOTAL_STEPS = NUM_EPOCHS * OPT_STEPS_PER_EPOCH

del _probe_loader

print(f"Micro batches / epoch: {MICRO_BATCHES_PER_EPOCH:,}")
print(f"Optimizer steps / epoch: {OPT_STEPS_PER_EPOCH:,}")
print(f"Planned optimizer steps: {PLANNED_TOTAL_STEPS:,}")
print(
    "Effective tokens / optimizer step:",
    MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE,
)


# ============================================================
# 5. Model config helpers
# ============================================================

def make_fresh_config():
    config = MiniLMConfig(
        block_size=BLOCK_SIZE,
        batch_size=MICRO_BATCH_SIZE,
        n_emb=N_EMB,
        n_layers=N_LAYERS,
        n_head=N_HEAD,
        drop_out=DROP_OUT,
        vocab_size=VOCAB_SIZE,
    )

    # Compatibility with an older dataclass implementation where these were
    # stored fields instead of @property values.
    if "head_size" in vars(config):
        config.head_size = config.n_emb // config.n_head
    if "intermediate_size" in vars(config):
        config.intermediate_size = 4 * config.n_emb

    if config.n_emb % config.n_head != 0:
        raise ValueError("n_emb must be divisible by n_head")

    return config


def build_model(config):
    model = GPT(config)

    if TIE_WORD_EMBEDDINGS:
        if not hasattr(model, "token_embedding") or not hasattr(model, "lm_head"):
            raise AttributeError(
                "GPT must expose token_embedding and lm_head to enable weight tying"
            )
        # Must happen before creating the optimizer and before loading a resume checkpoint.
        model.lm_head.weight = model.token_embedding.weight

    return model


# ============================================================
# 6. Checkpoint helpers
# ============================================================

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_training_checkpoint(
    model,
    optimizer,
    scaler,
    config,
    epoch,
    next_micro_batch,
    global_step,
    best_val_loss,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "epoch": epoch,
        "next_micro_batch": next_micro_batch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "planned_total_steps": PLANNED_TOTAL_STEPS,
        "rng_state": capture_rng_state(),
    }
    atomic_torch_save(payload, LATEST_CKPT)
    print(f"Checkpoint saved: {LATEST_CKPT} (step={global_step:,})")


def save_best_model(model, config, global_step, val_loss):
    payload = {
        "model": model.state_dict(),
        "config": config,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "global_step": global_step,
        "val_loss": val_loss,
    }
    atomic_torch_save(payload, BEST_MODEL)
    print(f"Best model updated: {BEST_MODEL}")


# ============================================================
# 7. Fresh start OR automatic resume
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if os.path.exists(LATEST_CKPT):
    print(f"Resuming from: {LATEST_CKPT}")
    ckpt = torch.load(
        LATEST_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    if ckpt.get("tokenizer_name") != TOKENIZER_NAME:
        raise RuntimeError(
            f"Checkpoint tokenizer={ckpt.get('tokenizer_name')} != {TOKENIZER_NAME}"
        )
    if ckpt.get("tokenizer_vocab_size") != VOCAB_SIZE:
        raise RuntimeError("Checkpoint tokenizer vocabulary does not match current tokenizer")
    if ckpt.get("planned_total_steps") != PLANNED_TOTAL_STEPS:
        raise RuntimeError(
            "Training plan changed since the checkpoint. Keep NUM_EPOCHS/data settings "
            "the same for an exact resume, or start a new training phase intentionally."
        )

    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["model"], strict=True)

    start_epoch = ckpt["epoch"]
    start_micro_batch = ckpt["next_micro_batch"]
    global_step = ckpt["global_step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
else:
    print("No checkpoint found; starting a NEW model from scratch.")
    config = make_fresh_config()
    model = build_model(config)

    start_epoch = 0
    start_micro_batch = 0
    global_step = 0
    best_val_loss = float("inf")
    ckpt = None

if config.vocab_size != VOCAB_SIZE:
    raise RuntimeError(
        f"Model config vocab_size={config.vocab_size}, tokenizer size={VOCAB_SIZE}"
    )
if config.block_size != BLOCK_SIZE:
    raise RuntimeError(
        f"Model config block_size={config.block_size}, script BLOCK_SIZE={BLOCK_SIZE}"
    )

model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
print("Config:", config)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MAX_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

if ckpt is not None:
    optimizer.load_state_dict(ckpt["optimizer"])


# ============================================================
# 8. AMP
# ============================================================

use_amp = device.type == "cuda"
use_scaler = False
amp_dtype = None

if use_amp:
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("AMP: BF16")
    else:
        amp_dtype = torch.float16
        use_scaler = True
        print("AMP: FP16 + GradScaler")
else:
    print("AMP: disabled")

scaler = torch.amp.GradScaler("cuda") if use_scaler else None

if ckpt is not None and scaler is not None and ckpt.get("scaler") is not None:
    scaler.load_state_dict(ckpt["scaler"])

# Restore dropout/random sampling state AFTER model/optimizer construction.
if ckpt is not None:
    restore_rng_state(ckpt.get("rng_state"))


# ============================================================
# 9. Warmup + cosine LR determined entirely from global_step
# ============================================================

def get_lr(step: int):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if PLANNED_TOTAL_STEPS <= WARMUP_STEPS:
        return MIN_LR

    progress = (step - WARMUP_STEPS) / (PLANNED_TOTAL_STEPS - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)

    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def set_optimizer_lr(lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# 10. Evaluation
# ============================================================

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    count = 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, targets=y)
        else:
            _, loss = model(x, targets=y)

        total_loss += loss.item()
        count += 1

        if count >= VAL_MAX_BATCHES:
            break

    model.train()
    return total_loss / max(count, 1)


# ============================================================
# 11. Training
# ============================================================

def train():
    global global_step, best_val_loss

    recent_losses = deque(maxlen=100)

    print(
        f"Start state: epoch={start_epoch}, "
        f"next_micro_batch={start_micro_batch:,}, "
        f"global_step={global_step:,}, "
        f"next_lr={get_lr(global_step):.8f}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loader = make_train_loader(epoch)

        # Only the first resumed epoch needs to skip already-completed batches.
        resume_batch = start_micro_batch if epoch == start_epoch else 0

        if resume_batch >= USABLE_MICRO_BATCHES:
            # Checkpoint was effectively at the end of this epoch.
            print(f"Epoch {epoch} already complete; moving to next epoch.")
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_raw_loss = 0.0
        accum_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= USABLE_MICRO_BATCHES:
                break

            # Rebuild exactly the same shuffled epoch and skip batches already trained.
            if batch_idx < resume_batch:
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, raw_loss = model(x, targets=y)
                    loss = raw_loss / GRAD_ACCUM_STEPS
            else:
                _, raw_loss = model(x, targets=y)
                loss = raw_loss / GRAD_ACCUM_STEPS

            accumulated_raw_loss += raw_loss.item()
            accum_count += 1

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer update only after a complete accumulation group.
            if accum_count == GRAD_ACCUM_STEPS:
                lr = get_lr(global_step)
                set_optimizer_lr(lr)

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                step_loss = accumulated_raw_loss / GRAD_ACCUM_STEPS
                recent_losses.append(step_loss)
                accumulated_raw_loss = 0.0
                accum_count = 0

                global_step += 1
                next_micro_batch = batch_idx + 1

                if global_step % LOG_EVERY == 0:
                    recent100 = sum(recent_losses) / len(recent_losses)
                    print(
                        f"Epoch={epoch} "
                        f"Step={global_step:,} "
                        f"MicroBatch={next_micro_batch:,}/{USABLE_MICRO_BATCHES:,} "
                        f"Loss={step_loss:.4f} "
                        f"Recent100={recent100:.4f} "
                        f"LR={lr:.8f}"
                    )

                if global_step % EVAL_EVERY == 0:
                    val_loss = evaluate()
                    ppl = math.exp(min(val_loss, 20.0))
                    print(
                        f"[Eval] Step={global_step:,} "
                        f"ValLoss={val_loss:.4f} PPL={ppl:.2f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_best_model(model, config, global_step, val_loss)

                if global_step % SAVE_EVERY == 0:
                    save_training_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        epoch=epoch,
                        next_micro_batch=next_micro_batch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )

        # End-of-epoch validation.
        val_loss = evaluate()
        ppl = math.exp(min(val_loss, 20.0))
        print(
            "\n========================================\n"
            f"Epoch {epoch} complete\n"
            f"Global step = {global_step:,}\n"
            f"Val Loss    = {val_loss:.4f}\n"
            f"PPL         = {ppl:.2f}\n"
            "========================================\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(model, config, global_step, val_loss)

        # Save a resume point that starts at the NEXT epoch.
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            epoch=epoch + 1,
            next_micro_batch=0,
            global_step=global_step,
            best_val_loss=best_val_loss,
        )

    print("Training finished.")


if __name__ == "__main__":
    train()
