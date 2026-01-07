"""
Token-Level Diffusion Model Training for Nmap Command Generation
FIXED & IMPROVED VERSION

Key improvements:
- Token-level CrossEntropy loss (discrete supervision)
- Token corruption (masking) for syntax repair
- Reduced diffusion steps (better stability)
- Clear comments everywhere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import math
import os
import time
from tqdm import tqdm

# AMP (mixed precision)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from token_level_tokenizer import NmapTokenLevelTokenizer

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Data
    "data_path": "nmap_dataset.json",
    "max_length": 32,

    # Training
    "batch_size": 64,
    "num_epochs": 70,              # ↓ was 200
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,

    # Model
    "d_model": 384,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 2048,       # ↑ slightly larger
    "dropout": 0.1,

    # Diffusion
    "num_timesteps": 200,          # ↓ was 500

    # Optimization
    "use_mixed_precision": True,
    "num_workers": 6,
    "pin_memory": True,

    # Checkpoints
    "save_dir": "./token_level_checkpoints",
    "save_every": 10,
}

# ============================================================================
# DATASET
# ============================================================================

class NmapTokenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "input_ids": torch.tensor(
                self.tokenizer.encode(item["input"], self.max_length),
                dtype=torch.long
            ),
            "output_ids": torch.tensor(
                self.tokenizer.encode(item["output"], self.max_length),
                dtype=torch.long
            ),
        }

# ============================================================================
# DIFFUSION SCHEDULE
# ============================================================================

class DiffusionSchedule:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        a = self.sqrt_alphas_cumprod[t][:, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return a * x_start + b * noise

# ============================================================================
# MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t):
        half = self.mlp[0].in_features // 2
        emb = torch.exp(
            torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class TokenDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, CONFIG["d_model"])
        self.pos_enc = PositionalEncoding(CONFIG["d_model"], CONFIG["max_length"])
        self.time_emb = TimestepEmbedding(CONFIG["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["nhead"],
            dim_feedforward=CONFIG["dim_feedforward"],
            dropout=CONFIG["dropout"],
            batch_first=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["nhead"],
            dim_feedforward=CONFIG["dim_feedforward"],
            dropout=CONFIG["dropout"],
            batch_first=False,
        )

        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=CONFIG["num_layers"] // 2
        )

        self.denoiser = nn.TransformerDecoder(
            decoder_layer, num_layers=CONFIG["num_layers"]
        )

        self.proj = nn.Linear(CONFIG["d_model"], CONFIG["d_model"])

    def forward(self, noisy_emb, t, context_ids, context_mask):
        time = self.time_emb(t).unsqueeze(1)
        noisy_emb = noisy_emb + time

        tgt = self.pos_enc(noisy_emb.transpose(0, 1))

        ctx = self.token_embedding(context_ids).transpose(0, 1)
        ctx = self.pos_enc(ctx)
        ctx = self.context_encoder(ctx, src_key_padding_mask=context_mask)

        out = self.denoiser(tgt, ctx, memory_key_padding_mask=context_mask)
        return self.proj(out.transpose(0, 1))

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, tokenizer, schedule):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.schedule = schedule

        # 🔥 MOVE SCHEDULE TENSORS TO DEVICE (FIX)
        for name, value in vars(self.schedule).items():
            if torch.is_tensor(value):
                setattr(self.schedule, name, value.to(self.device))

        self.use_amp = CONFIG["use_mixed_precision"] and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

    def corrupt_tokens(self, ids, p=0.15):
        """
        Semantic token corruption:
        - flags → <UNK_FLAG>
        - arguments → <UNK_ARG>
        - targets → <UNK_TARGET>
        """

        pad_id = self.tokenizer.token2idx[self.tokenizer.PAD]

        unk_flag = self.tokenizer.token2idx[self.tokenizer.UNK_FLAG]
        unk_arg = self.tokenizer.token2idx[self.tokenizer.UNK_ARG]
        unk_target = self.tokenizer.token2idx[self.tokenizer.UNK_TARGET]

        out = ids.clone()
        rand = torch.rand_like(ids.float())

        for i in range(ids.size(0)):
            for j in range(ids.size(1)):
                if rand[i, j] < p and ids[i, j] != pad_id:
                    tok = self.tokenizer.idx2token[int(ids[i, j])]

                    if tok.startswith("-"):
                        out[i, j] = unk_flag
                    elif tok in {self.tokenizer.IP, self.tokenizer.IP_CIDR, self.tokenizer.TARGET}:
                        out[i, j] = unk_target
                    else:
                        out[i, j] = unk_arg

        return out

    def train_step(self, batch, opt):
        self.model.train()

        inp = batch["input_ids"].to(self.device)
        out = batch["output_ids"].to(self.device)

        # ---- token corruption ----
        corrupted = self.corrupt_tokens(out)

        clean_emb = self.model.token_embedding(out)
        noisy_emb = self.model.token_embedding(corrupted)

        noise = torch.randn_like(clean_emb)
        t = torch.randint(0, self.schedule.num_timesteps, (out.size(0),), device=self.device)

        noisy_emb = self.schedule.q_sample(noisy_emb, t, noise)

        mask = inp == self.tokenizer.token2idx[self.tokenizer.PAD]

        if self.use_amp:
            try:
                ctx = autocast(device_type="cuda")
            except TypeError:
                ctx = autocast()
        else:
            ctx = torch.no_grad()  # dummy context

        with ctx:
            pred = self.model(noisy_emb, t, inp, mask)

            # ---- embedding loss ----
            mse = F.mse_loss(pred, clean_emb)

            # ---- token loss ----
            logits = torch.matmul(pred, self.model.token_embedding.weight.T)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                out.view(-1),
                ignore_index=self.tokenizer.token2idx[self.tokenizer.PAD],
            )

            loss = mse + 0.5 * ce

        opt.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["grad_clip"])
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            opt.step()

        return loss.item()

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    with open(CONFIG["data_path"]) as f:
        data = json.load(f)

    tokenizer = NmapTokenLevelTokenizer()
    tokenizer.build_vocab([d["output"] for d in data])
    tokenizer.save_vocab(os.path.join(CONFIG["save_dir"], "vocab.json"))

    dataset = NmapTokenDataset(data, tokenizer, CONFIG["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = TokenDiffusionTransformer(tokenizer.vocab_size)
    schedule = DiffusionSchedule(CONFIG["num_timesteps"])
    trainer = Trainer(model, tokenizer, schedule)

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["num_epochs"]):
        losses = []
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}"):
            losses.append(trainer.train_step(batch, opt))

        avg = sum(losses) / len(losses)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

        if (epoch + 1) % CONFIG["save_every"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG["save_dir"], f"model_{epoch+1}.pt"),
            )

    print("Training complete.")

if __name__ == "__main__":
    main()
