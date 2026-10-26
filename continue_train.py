import os
import math
import shutil
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from mini_lm import GPT


# ============================================================
# 0. Resume / dataset config
# ============================================================

RESUME_CHECKPOINT = "./checkpoints_continue/step_50000.pt"

CHECKPOINT_DIR = "./checkpoints_continue"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 中间训练状态始终覆盖这个文件，避免每 5000 step 留一个 2~3GB checkpoint
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pt")

TOKEN_DIR = "./processed_chinese_corpus"
os.makedirs(TOKEN_DIR, exist_ok=True)

TRAIN_BIN = os.path.join(TOKEN_DIR, "train_tokens.bin")
VAL_BIN = os.path.join(TOKEN_DIR, "val_tokens.bin")

DATASET_NAME = "richarddzh/chinese-small-lm-corpus"
VAL_DOCS = 10_000
TOKENIZE_BATCH_SIZE = 64

# DataLoader batch size。模型结构从 checkpoint 的 config 恢复，不重新定义。
TRAIN_BATCH_SIZE = 12
NUM_WORKERS = 4

LOG_EVERY = 100
SAVE_EVERY = 5000
VAL_MAX_BATCHES = 300

# 仅在 checkpoint 缺少 optimizer/scheduler 时作为 fallback。
FALLBACK_LR = 1e-4
FALLBACK_MIN_LR = 1e-5
WEIGHT_DECAY = 0.1

# 若 token bin 已经存在则直接复用。
FORCE_REBUILD_TOKENS = False


# ============================================================
# 1. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 我们只使用 GPT-2 tokenizer，不使用原 GPT-2 模型的 1024 context 限制。
tokenizer.model_max_length = 10**9

print("Tokenizer vocab size:", len(tokenizer))
print("EOS token id:", tokenizer.eos_token_id)


# ============================================================
# 2. Pre-tokenize HuggingFace corpus -> uint16 binary files
# ============================================================

def write_token_stream(dataset_iter, output_path, tokenizer, batch_size=64):
    token_count = 0
    doc_count = 0
    texts = []

    with open(output_path, "wb") as f:

        def process_batch(batch_texts):
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
                ids.append(tokenizer.eos_token_id)

                # GPT-2 vocab_size=50257 < 65536，所以 uint16 足够。
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(f)
                token_count += len(ids)

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
                process_batch(texts)
                texts.clear()

                if doc_count % 10_000 == 0:
                    print(
                        f"Processed docs: {doc_count:,}, "
                        f"tokens: {token_count / 1e6:.2f}M"
                    )

        process_batch(texts)

    print(
        f"\nFinished: {output_path}\n"
        f"Documents: {doc_count:,}\n"
        f"Tokens: {token_count / 1e6:.2f}M\n"
    )


def prepare_dataset():
    if FORCE_REBUILD_TOKENS:
        for path in (TRAIN_BIN, VAL_BIN):
            if os.path.exists(path):
                os.remove(path)

    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        print("Token files already exist. Skip tokenization.")
        return

    print("\nLoading Chinese corpus...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # streaming 模式只能做 buffer shuffle；这里只用于打散 validation/train 的文档来源。
    dataset = dataset.shuffle(seed=42, buffer_size=20_000)

    val_stream = dataset.take(VAL_DOCS)
    train_stream = dataset.skip(VAL_DOCS)

    print("\nBuilding validation tokens...")
    write_token_stream(
        val_stream,
        VAL_BIN,
        tokenizer,
        TOKENIZE_BATCH_SIZE,
    )

    print("\nBuilding training tokens...")
    write_token_stream(
        train_stream,
        TRAIN_BIN,
        tokenizer,
        TOKENIZE_BATCH_SIZE,
    )


# ============================================================
# 3. mmap dataset
# ============================================================

class TokenBinDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r",
        )
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size

        chunk = self.data[
            start : start + self.block_size + 1
        ]

        # memmap 是只读的，copy 后再交给 torch。
        chunk = np.array(chunk, dtype=np.int64, copy=True)
        chunk = torch.from_numpy(chunk)

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ============================================================
# 4. Safe/atomic checkpoint save
# ============================================================

def atomic_torch_save(obj, path):
    """先写 .tmp，成功后再原子替换，避免 latest.pt 被半写坏。"""
    tmp_path = path + ".tmp"

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def print_disk_space(path="."):
    total, used, free = shutil.disk_usage(path)
    print(
        f"Disk: used={used / 1024**3:.1f} GiB, "
        f"free={free / 1024**3:.1f} GiB, "
        f"total={total / 1024**3:.1f} GiB"
    )


# ============================================================
# 5. Main
# ============================================================

def main():
    prepare_dataset()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.set_float32_matmul_precision("high")

    # --------------------------------------------------------
    # Load step_50000.pt
    # --------------------------------------------------------
    if not os.path.exists(RESUME_CHECKPOINT):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {RESUME_CHECKPOINT}"
        )

    print("\nLoading resume checkpoint:", RESUME_CHECKPOINT)

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain at least 'model' and 'config'."
        )

    config = checkpoint["config"]
    start_global_step = int(checkpoint.get("global_step", 50_000))
    start_epoch = int(checkpoint.get("epoch", 0))

    print("\nLoaded config:")
    print(config)
    print("Resume epoch:", start_epoch)
    print("Resume global step:", start_global_step)

    assert config.vocab_size == len(tokenizer), (
        f"Checkpoint vocab={config.vocab_size}, "
        f"tokenizer vocab={len(tokenizer)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_dataset = TokenBinDataset(TRAIN_BIN, config.block_size)
    val_dataset = TokenBinDataset(VAL_BIN, config.block_size)

    print("\nTrain tokens:", f"{len(train_dataset.data):,}")
    print("Val tokens:", f"{len(val_dataset.data):,}")
    print("Train blocks:", f"{len(train_dataset):,}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    print("Steps per full data pass:", f"{len(train_loader):,}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = GPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # --------------------------------------------------------
    # Optimizer: restore Adam m/v and optimizer step
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FALLBACK_LR,
        weight_decay=WEIGHT_DECAY,
    )

    if "optimizer" in checkpoint:
        print("Loading optimizer state...")
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("WARNING: checkpoint has no optimizer state; using a new AdamW.")

    # --------------------------------------------------------
    # Scheduler: restore exact cosine state from step_50000
    # --------------------------------------------------------
    if "scheduler" in checkpoint:
        scheduler_state = checkpoint["scheduler"]
        saved_t_max = int(
            scheduler_state.get(
                "T_max",
                max(start_global_step + len(train_loader), 1),
            )
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=saved_t_max,
            eta_min=float(scheduler_state.get("eta_min", FALLBACK_MIN_LR)),
        )
        scheduler.load_state_dict(scheduler_state)

        # 原训练计划的总 optimizer step 数。
        target_total_steps = saved_t_max
        print("Loading scheduler state...")
    else:
        # 只有在老 checkpoint 没保存 scheduler 时才走这里。
        remaining_steps = 2 * len(train_loader)
        target_total_steps = start_global_step + remaining_steps

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=target_total_steps,
            eta_min=FALLBACK_MIN_LR,
            last_epoch=start_global_step - 1,
        )

        print("WARNING: checkpoint has no scheduler state; created fallback schedule.")

    # 防止历史 checkpoint 中 global_step 与 scheduler.last_epoch 差 1 时难以察觉。
    print("Scheduler last_epoch:", scheduler.last_epoch)
    print("Target total steps:", f"{target_total_steps:,}")
    print("Current LR:", optimizer.param_groups[0]["lr"])

    if start_global_step >= target_total_steps:
        print(
            "Checkpoint global_step already reached scheduler T_max. "
            "Nothing left to train under the original schedule."
        )
        return

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------
    use_amp = device.type == "cuda"

    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
        print("AMP: BF16")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        print("AMP: FP16")

        # 新版 checkpoint 会保存 scaler；旧 step_50000.pt 没有也没关系。
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        amp_dtype = None
        scaler = None
        print("AMP: disabled")

    # --------------------------------------------------------
    # Save helpers
    # --------------------------------------------------------
    def save_training_checkpoint(epoch, global_step, path):
        print_disk_space(CHECKPOINT_DIR)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }

        atomic_torch_save(state, path)
        print(f"\nCheckpoint saved: {path}\n")

    def save_model_only(epoch, global_step):
        path = os.path.join(
            CHECKPOINT_DIR,
            f"model_epoch_{epoch}_step_{global_step}.pt",
        )

        state = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }

        atomic_torch_save(state, path)
        print(f"Model-only checkpoint saved: {path}")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate():
        model.eval()

        total_loss = 0.0
        count = 0

        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)

            total_loss += loss.item()
            count += 1

            if count >= VAL_MAX_BATCHES:
                break

        model.train()
        return total_loss / max(count, 1)

    # --------------------------------------------------------
    # Continue training from step 50000
    # --------------------------------------------------------
    global_step = start_global_step
    epoch = start_epoch

    recent_losses = deque(maxlen=100)

    print(
        f"\nContinue training from global_step={global_step:,} "
        f"until global_step={target_total_steps:,}."
    )
    print(
        "Note: DataLoader shuffle order is recreated after restart; "
        "model/optimizer/scheduler steps are continuous, but the exact old "
        "batch permutation is not reproduced.\n"
    )

    while global_step < target_total_steps:
        model.train()
        pass_loss = 0.0
        pass_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if global_step >= target_total_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

            scheduler.step()
            global_step += 1

            loss_value = loss.item()
            pass_loss += loss_value
            pass_batches += 1
            recent_losses.append(loss_value)

            if global_step % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]["lr"]
                recent_avg = sum(recent_losses) / len(recent_losses)

                print(
                    f"Epoch={epoch} "
                    f"Step={global_step:,} "
                    f"Batch={batch_idx:,} "
                    f"Loss={loss_value:.4f} "
                    f"Recent100={recent_avg:.4f} "
                    f"LR={lr:.8f}"
                )

            # 永远覆盖 latest.pt，不再生成 step_55000.pt / step_60000.pt ...
            if global_step % SAVE_EVERY == 0:
                save_training_checkpoint(
                    epoch,
                    global_step,
                    LATEST_CHECKPOINT,
                )

        val_loss = evaluate()
        train_loss = pass_loss / max(pass_batches, 1)

        print(
            "\n"
            "========================================\n"
            f"Data pass / Epoch {epoch} completed\n"
            f"Current global step = {global_step:,}\n"
            f"Train Loss          = {train_loss:.4f}\n"
            f"Val Loss            = {val_loss:.4f}\n"
            f"PPL                 = {math.exp(min(val_loss, 20)):.2f}\n"
            "========================================\n"
        )

        # 每轮数据结束保留一个较小的 model-only 文件。
        save_model_only(epoch, global_step)

        # 同时更新可继续恢复训练的 latest.pt。
        save_training_checkpoint(
            epoch,
            global_step,
            LATEST_CHECKPOINT,
        )

        epoch += 1

    print(
        f"\nTraining schedule completed at global_step={global_step:,}."
    )


if __name__ == "__main__":
    main()
