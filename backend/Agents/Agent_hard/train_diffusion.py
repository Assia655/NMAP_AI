import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from diffusers import DDPMScheduler
import json
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 16  # Reduced for 8GB VRAM
LR = 1e-4
EPOCHS = 10
MAX_LEN = 64  # Max length of Nmap command
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 1. DATASET CLASS ---
class NmapDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=64):
        with open(json_file, 'r') as f:
            # Handle the JSON format (list of objects)
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Input: User Query (Condition)
        cond_encoding = self.tokenizer(
            item['input'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # Output: Nmap Command (Target)
        target_encoding = self.tokenizer(
            item['output'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'cond_ids': cond_encoding['input_ids'].squeeze(0),
            'cond_mask': cond_encoding['attention_mask'].squeeze(0),
            'target_ids': target_encoding['input_ids'].squeeze(0)
        }


# --- 2. THE DIFFUSION MODEL ---
class TextDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')  # Context Encoder
        self.decoder = BertModel(self.config)  # The Denoiser (predicts clean embeddings)

        # Output projection back to vocabulary size
        self.out_proj = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        # Time embedding (to tell the model which step t we are at)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

    def get_embeddings(self, input_ids):
        return self.decoder.embeddings.word_embeddings(input_ids)

    def forward(self, noisy_embeddings, timesteps, cond_ids, cond_mask):
        # 1. Encode the Condition (User Query)
        cond_output = self.encoder(input_ids=cond_ids, attention_mask=cond_mask)
        cond_feats = cond_output.last_hidden_state  # [Batch, Seq, Dim]

        # 2. Embed Time
        t_emb = self.time_embed(timesteps.unsqueeze(-1).float())  # [Batch, Dim]
        t_emb = t_emb.unsqueeze(1)  # Broadcast to seq len

        # 3. Denoising Step
        # We concatenate condition + noisy_input to guide generation
        # (Simplified Cross-Attention via concatenation for this demo)
        combined_input = noisy_embeddings + t_emb

        # Pass through Decoder (Denoiser)
        # We use the condition as 'encoder_hidden_states' for cross-attention
        outputs = self.decoder(
            inputs_embeds=combined_input,
            encoder_hidden_states=cond_feats,
            encoder_attention_mask=cond_mask
        )

        # Predict the CLEAN embeddings (x_0) directly
        pred_embeddings = outputs.last_hidden_state
        return pred_embeddings


# --- 3. TRAINING LOOP ---
def train():
    print(f"[*] Training on {DEVICE} with Mixed Precision")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = NmapDataset('train.json', tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TextDiffusion().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    scaler = torch.cuda.amp.GradScaler()  # For Mixed Precision (FP16)

    model.train()

    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in loop:
            cond_ids = batch['cond_ids'].to(DEVICE)
            cond_mask = batch['cond_mask'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)

            # A. Prepare Inputs
            # Get the "clean" embeddings of the Nmap command
            with torch.no_grad():
                clean_embeddings = model.get_embeddings(target_ids)

            # B. Add Noise (Forward Diffusion)
            noise = torch.randn_like(clean_embeddings).to(DEVICE)
            bsz = clean_embeddings.shape[0]
            # Sample random timesteps
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()

            # Add noise using the scheduler
            noisy_embeddings = scheduler.add_noise(clean_embeddings, noise, timesteps)

            # C. Model Prediction
            with torch.cuda.amp.autocast():  # FP16 Context
                pred_embeddings = model(noisy_embeddings, timesteps, cond_ids, cond_mask)

                # Loss: Distance between Predicted Embeddings and Clean Embeddings
                # (Diffusion-LM usually predicts x_0 directly)
                loss = nn.MSELoss()(pred_embeddings, clean_embeddings)

            # D. Backprop
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

    # Save the specialized agent
    torch.save(model.state_dict(), "hard_agent_diffusion.pth")
    print("[+] Model Saved!")


if __name__ == "__main__":
    train()