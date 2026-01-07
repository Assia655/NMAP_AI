import torch
import torch.nn.functional as F
from train_model import TokenDiffusionTransformer, DiffusionSchedule, CONFIG
from token_level_tokenizer import NmapTokenLevelTokenizer

# ============================================================
# SETUP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./token_level_checkpoints/model_70.pt"

# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------
tokenizer = NmapTokenLevelTokenizer()
tokenizer.load_vocab("./token_level_checkpoints/vocab.json")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model = TokenDiffusionTransformer(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# ------------------------------------------------------------
# Load diffusion schedule
# ------------------------------------------------------------
schedule = DiffusionSchedule(CONFIG["num_timesteps"])
for k, v in vars(schedule).items():
    if torch.is_tensor(v):
        setattr(schedule, k, v.to(device))

# ============================================================
# POST-PROCESSING HELPERS
# ============================================================

def deduplicate(tokens):
    """Remove repeated tokens while preserving order"""
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def enforce_minimum_syntax(tokens):
    """
    Enforce minimal Nmap command structure
    """
    # Ensure command starts with nmap
    if not tokens or tokens[0] != "nmap":
        tokens = ["nmap"] + tokens

    # Only one port specification
    port_tokens = [
        t for t in tokens
        if t in {tokenizer.PORT_ALL, tokenizer.PORT_SINGLE,
                 tokenizer.PORT_LIST, tokenizer.PORT_RANGE, "-p-"}
    ]

    if len(port_tokens) > 1:
        keep = port_tokens[0]
        tokens = [t for t in tokens if t not in port_tokens or t == keep]

    return tokens


def topk_sample(similarity, k=5):
    """
    Top-k sampling instead of argmax to reduce mode collapse
    """
    vals, idxs = similarity.topk(k, dim=-1)
    choice = torch.randint(0, k, (idxs.size(0),), device=idxs.device)
    return idxs[torch.arange(idxs.size(0)), choice]

# ============================================================
# GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate(prompt, steps=40):
    # Encode input
    input_ids = torch.tensor(
        tokenizer.encode(prompt, CONFIG["max_length"]),
        device=device
    ).unsqueeze(0)

    mask = input_ids == tokenizer.token2idx[tokenizer.PAD]

    # Start from noise
    x = torch.randn(
        1, CONFIG["max_length"], CONFIG["d_model"], device=device
    )

    # Timesteps
    timesteps = list(range(schedule.num_timesteps))[::-1]
    timesteps = timesteps[:: max(1, len(timesteps) // steps)]

    for t in timesteps:
        t = torch.tensor([t], device=device)
        x = model(x, t, input_ids, mask)

    # Embedding → token similarity
    sim = torch.matmul(
        F.normalize(x, dim=-1),
        F.normalize(model.token_embedding.weight, dim=-1).T
    ).squeeze(0)

    # Sample tokens
    token_ids = topk_sample(sim, k=5).tolist()
    tokens = [tokenizer.idx2token[i] for i in token_ids]

    # Post-processing
    tokens = deduplicate(tokens)
    tokens = enforce_minimum_syntax(tokens)

    return tokenizer.decode(
        [tokenizer.token2idx[t] for t in tokens],
        skip_special=True
    )

# ============================================================
# TEST CASES
# ============================================================

tests = [
    "Scan all UDP ports on 192.168.1.1",
    "Pingless scan on ports 22 and 8080",
    "Fast scan with OS detection",
    "Scan ports 1-1000 with service detection",
    "Quick ping scan on 10.0.0.5",
]

print("\n================= MODEL TEST =================\n")
for t in tests:
    print("Input :", t)
    print("Output:", generate(t))
    print("-" * 55)
