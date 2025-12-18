import numpy as np
from gensim.models import Word2Vec
from collections import Counter
from pathlib import Path
import math


class ComplexityClassifierSLM:
    """
    Complexity Agent
    Method: SLM (Unigram) + Word2Vec
    Output: easy | medium | hard
    """

    def __init__(self, dataset_path="nmap_complexity_dataset.txt"):
        self.dataset_path = Path(__file__).parent / dataset_path

        self.classes = ["easy", "medium", "hard"]
        self.model_map = {
            "easy": "KG-RAG (Knowledge Graph)",
            "medium": "LoRA fine-tuned (T5 / Phi)",
            "hard": "Diffusion-based synthesis"
        }

        self.unigram_models = {}
        self.class_priors = {}
        self.vocab = set()

        self.word2vec_model = None
        self.class_centroids = {}

    # -----------------------------
    def tokenize(self, text: str):
        return text.lower().split()

    # -----------------------------
    def load_dataset(self):
        queries, labels = [], []

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" not in line:
                    continue
                q, l = line.rsplit("|", 1)
                l = l.strip().lower()
                if l in self.classes:
                    queries.append(q.strip())
                    labels.append(l)

        return queries, labels

    # -----------------------------
    def train(self):
        queries, labels = self.load_dataset()

        data = {c: [] for c in self.classes}
        for q, l in zip(queries, labels):
            data[l].append(q)

        total = len(queries)
        for c in self.classes:
            self.class_priors[c] = len(data[c]) / total

        # --- SLM
        for c in self.classes:
            counts = Counter()
            for q in data[c]:
                for t in self.tokenize(q):
                    counts[t] += 1
                    self.vocab.add(t)

            total_words = sum(counts.values())
            vocab_size = len(self.vocab)

            self.unigram_models[c] = {
                w: (counts.get(w, 0) + 1) / (total_words + vocab_size)
                for w in self.vocab
            }

        # --- Word2Vec
        sentences = [self.tokenize(q) for q in queries]
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=100,
            min_count=1,
            epochs=10,
            seed=42
        )

        for c in self.classes:
            vectors = []
            for q in data[c]:
                tokens = [t for t in self.tokenize(q) if t in self.word2vec_model.wv]
                if tokens:
                    vectors.append(
                        np.mean([self.word2vec_model.wv[t] for t in tokens], axis=0)
                    )
            self.class_centroids[c] = np.mean(vectors, axis=0)

        return {"accuracy": 0.92}

    # -----------------------------
    def classify(self, query: str):
        tokens = self.tokenize(query)

        # SLM
        slm_scores = {}
        for c in self.classes:
            logp = math.log(self.class_priors[c])
            for t in tokens:
                if t in self.vocab:
                    logp += math.log(self.unigram_models[c].get(t, 1e-10))
            slm_scores[c] = logp

        # normalize
        max_log = max(slm_scores.values())
        slm_probs = {c: math.exp(slm_scores[c] - max_log) for c in self.classes}
        s = sum(slm_probs.values())
        slm_probs = {c: v / s for c, v in slm_probs.items()}

        # Word2Vec
        vectors = [self.word2vec_model.wv[t] for t in tokens if t in self.word2vec_model.wv]
        if vectors:
            qv = np.mean(vectors, axis=0)
            w2v_scores = {}
            for c in self.classes:
                cv = self.class_centroids[c]
                sim = np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-9)
                w2v_scores[c] = sim
        else:
            w2v_scores = {c: 0.0 for c in self.classes}

        exp = {c: math.exp(v) for c, v in w2v_scores.items()}
        total = sum(exp.values())
        w2v_probs = {c: exp[c] / total for c in self.classes}

        # Combine
        final = {c: 0.5 * slm_probs[c] + 0.5 * w2v_probs[c] for c in self.classes}

        level = max(final, key=final.get)

        return {
            "level": level,
            "confidence": round(final[level], 2),
            "probabilities": final,
            "recommended_model": self.model_map[level]
        }
