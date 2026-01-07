"""
Diffusion-Optimized Token-Level Nmap Tokenizer
Designed specifically for embedding-space diffusion models
"""

import re
import json
from typing import List, Dict
from collections import Counter


class NmapTokenLevelTokenizer:
    def __init__(self):
        # ===== Special tokens =====
        self.PAD = "<PAD>"
        self.START = "<START>"
        self.END = "<END>"

        # Typed unknowns (VERY important for diffusion)
        self.UNK_FLAG = "<UNK_FLAG>"
        self.UNK_ARG = "<UNK_ARG>"
        self.UNK_TARGET = "<UNK_TARGET>"

        # Semantic placeholders
        self.IP = "<IP>"
        self.IP_CIDR = "<IP_CIDR>"
        self.PORT_SINGLE = "<PORT>"
        self.PORT_LIST = "<PORT_LIST>"
        self.PORT_RANGE = "<PORT_RANGE>"
        self.PORT_ALL = "<PORT_ALL>"
        self.TARGET = "<TARGET>"

        self.special_tokens = [
            self.PAD, self.START, self.END,
            self.UNK_FLAG, self.UNK_ARG, self.UNK_TARGET,
            self.IP, self.IP_CIDR,
            self.PORT_SINGLE, self.PORT_LIST, self.PORT_RANGE, self.PORT_ALL,
            self.TARGET
        ]

        self.token2idx = {}
        self.idx2token = {}
        self.vocab_size = 0

    # =========================================================================
    # Vocabulary
    # =========================================================================

    def build_vocab(self, commands: List[str], min_freq: int = 1):
        counter = Counter()

        for cmd in commands:
            counter.update(self._smart_tokenize(cmd))

        # Add special tokens first
        for idx, tok in enumerate(self.special_tokens):
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

        idx = len(self.special_tokens)

        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in self.token2idx:
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok
                idx += 1

        self.vocab_size = len(self.token2idx)

        print(f"✓ Vocabulary built: {self.vocab_size} tokens")

    # =========================================================================
    # Tokenization (DIFFUSION SAFE)
    # =========================================================================

    def _smart_tokenize(self, command: str) -> List[str]:
        tokens = []
        parts = re.sub(r"\s+", " ", command.strip()).split()

        i = 0
        while i < len(parts):
            part = parts[i]

            # ---- Flags with arguments (bind them!) ----
            if part in {"-p", "--ports"} and i + 1 < len(parts):
                port_spec = parts[i + 1]
                tokens.append(self._normalize_ports(port_spec))
                i += 2
                continue

            if part in {"--script", "--scripts"} and i + 1 < len(parts):
                tokens.append(f"<SCRIPT:{parts[i + 1]}>")
                i += 2
                continue

            if part == "--script-args" and i + 1 < len(parts):
                tokens.append("<SCRIPT_ARGS>")
                i += 2
                continue

            # ---- Flags (atomic) ----
            if part.startswith("-"):
                tokens.append(part)
                i += 1
                continue

            # ---- IPs ----
            if re.fullmatch(r"\d+\.\d+\.\d+\.\d+/\d+", part):
                tokens.append(self.IP_CIDR)
                i += 1
                continue

            if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", part):
                tokens.append(self.IP)
                i += 1
                continue

            # ---- Targets ----
            if part.lower() in {"target", "targets"}:
                tokens.append(self.TARGET)
                i += 1
                continue

            # ---- Fallback ----
            tokens.append(part)
            i += 1

        return tokens

    def _normalize_ports(self, spec: str) -> str:
        if spec == "-":
            return self.PORT_ALL
        if "," in spec:
            return self.PORT_LIST
        if "-" in spec:
            return self.PORT_RANGE
        if spec.isdigit():
            return self.PORT_SINGLE
        return self.UNK_ARG

    # =========================================================================
    # Encode / Decode
    # =========================================================================

    def encode(self, command: str, max_length: int) -> List[int]:
        tokens = [self.START] + self._smart_tokenize(command) + [self.END]

        ids = []
        for tok in tokens:
            ids.append(self.token2idx.get(tok, self.token2idx[self.UNK_ARG]))

        if len(ids) < max_length:
            ids += [self.token2idx[self.PAD]] * (max_length - len(ids))
        else:
            ids = ids[: max_length - 1] + [self.token2idx[self.END]]

        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in token_ids:
            tok = self.idx2token.get(i, "")
            if skip_special and tok.startswith("<"):
                continue
            tokens.append(tok)
        return " ".join(tokens)

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save_vocab(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "token2idx": self.token2idx,
                    "idx2token": self.idx2token,
                    "special_tokens": self.special_tokens,
                },
                f,
                indent=2,
            )

    def load_vocab(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.token2idx = data["token2idx"]
        self.idx2token = {int(k): v for k, v in data["idx2token"].items()}
        self.special_tokens = data["special_tokens"]
        self.vocab_size = len(self.token2idx)
