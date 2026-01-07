import sys
import types
import importlib
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

nlp = None


def _install_fake_torch():
    """Install a lightweight torch stub to avoid DLL errors when torch is broken."""
    if "torch" in sys.modules:
        return

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "0.0.0"
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 0, amp=types.SimpleNamespace(common=types.SimpleNamespace(amp_definitely_not_available=lambda: True)))
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_built=lambda: False, is_available=lambda: False))
    fake_utils = types.ModuleType("torch.utils")
    fake_dlpack = types.ModuleType("torch.utils.dlpack")
    fake_utils.dlpack = fake_dlpack
    fake_torch.utils = fake_utils

    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.dlpack"] = fake_dlpack


def _load_spacy_pipeline():
    """Load spaCy while tolerating broken torch installations."""
    global nlp
    try:
        spacy = importlib.import_module("spacy")  # type: ignore
    except Exception as exc:
        print(f"[Comprehension] spaCy import failed: {exc}")
        print("[Comprehension] Attempting fallback with torch stub...")
        _install_fake_torch()
        try:
            spacy = importlib.import_module("spacy")  # type: ignore
            print("[Comprehension] spaCy imported with torch stub.")
        except Exception as exc2:
            print(f"[Comprehension] spaCy still failing to import: {exc2}")
            print("[Comprehension] Ensure spaCy is installed: pip install spacy")
            print("[Comprehension] Download the model: python -m spacy download en_core_web_sm")
            return

    try:
        nlp = spacy.load("en_core_web_sm")
        print("[Comprehension] spaCy English model loaded.")
    except Exception as exc:
        print(f"[Comprehension] Unable to load spaCy model: {exc}")
        print("[Comprehension] Retrying with torch stub...")
        _install_fake_torch()
        try:
            nlp = spacy.load("en_core_web_sm")
            print("[Comprehension] spaCy English model loaded (torch stub).")
        except Exception as exc2:
            print(f"[Comprehension] Still unable to load spaCy model: {exc2}")
            print("[Comprehension] Install the model with: python -m spacy download en_core_web_sm")
            nlp = None


_load_spacy_pipeline()


class NMAPEmbeddingAgent:
    """
    Agent NMAP avec embeddings spaCy.
    Simple, rapide, et fiable.
    """

    def __init__(self, corpus_file: str = "nmap_domain.txt"):
        if nlp is None:
            raise RuntimeError(
                "spaCy pipeline not initialized. Install spaCy and the 'en_core_web_sm' model."
            )

        self.nlp = nlp
        self.corpus_file = corpus_file
        self.corpus_documents: List[str] = []

        # Mots clairement hors contexte
        self.out_of_context_words = {
            "document",
            "fichier",
            "pdf",
            "word",
            "excel",
            "video",
            "image",
            "photo",
            "music",
            "email",
            "movie",
            "book",
            "car",
            "house",
            "sport",
            "play",
            "game",
            "film",
            "audio",
            "sound",
            "voice",
            "speech",
            "lire",
            "regarder",
            "livre",
            "ouvrir",
            "musique",
            "video",
            "cinema",
            "serie",
            "podcast",
            "streaming",
            "jouer",
            "gamer",
            "console",
            "jeux",
            "imprimante",
            "passeport",
        }

        # Mots-clés NMAP
        self.nmap_keywords = {
            "scan",
            "port",
            "host",
            "service",
            "version",
            "os",
            "detect",
            "discover",
            "enumerate",
            "script",
            "nse",
            "map",
            "network",
            "probe",
            "syn",
            "tcp",
            "udp",
            "icmp",
            "ping",
            "trace",
            "route",
            "firewall",
            "vulnerability",
            "exploit",
            "cve",
            "banner",
            "target",
            "machine",
            "address",
            "ip",
            "subnet",
            "cidr",
            "ipv6",
            "timing",
            "aggressive",
            "stealth",
            "evade",
            "bypass",
            "fragment",
            "xml",
            "audit",
            "reconnaissance",
            "enumeration",
            "topologie",
            "infrastructure",
        }

        print("Initializing NMAP agent with spaCy...")
        self._load_corpus()
        self._compute_nmap_embedding()
        print("Agent ready.\n")

    def _load_corpus(self):
        """Charger le corpus"""
        try:
            with open(self.corpus_file, "r", encoding="utf-8") as f:
                self.corpus_documents = [
                    line.strip() for line in f if line.strip() and not line.startswith("#")
                ]
            print(f"Corpus charge: {len(self.corpus_documents)} documents")
        except FileNotFoundError:
            print(f"Fichier '{self.corpus_file}' non trouve")
            raise

    def _compute_nmap_embedding(self):
        """Calculer l'embedding representant le domaine NMAP"""
        print("Calcul de l'embedding NMAP...")

        nmap_docs = []
        for doc_text in self.corpus_documents[:50]:  # Utiliser les 50 premiers
            doc = self.nlp(doc_text)
            if doc.has_vector:
                nmap_docs.append(doc.vector)

        if nmap_docs:
            self.nmap_embedding = np.mean(nmap_docs, axis=0)
        else:
            vector_dim = self.nlp.vocab.vectors_length or 96
            self.nmap_embedding = np.zeros(vector_dim)

        print(f"Embedding NMAP calcule (dimension: {len(self.nmap_embedding)})")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculer la similarite cosinus"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        # Eviter les vecteurs nuls
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        return float(dot_product / (norm_vec1 * norm_vec2))

    def analyze_query(self, query: str) -> Dict:
        """Analyser une query avec spaCy embeddings"""

        # Parser la query avec spaCy
        doc = self.nlp(query.lower())

        if not doc.has_vector:
            return {
                "tokens": [token.text for token in doc],
                "avg_similarity": 0.0,
                "nmap_keywords_found": [],
                "out_of_context": [],
                "query_length": len(doc),
            }

        # Similarite globale
        avg_similarity = self.cosine_similarity(doc.vector, self.nmap_embedding)

        # Mots NMAP et hors contexte
        nmap_keywords_found = []
        out_of_context = []

        for token in doc:
            word = token.text.lower()
            if word in self.nmap_keywords:
                nmap_keywords_found.append(word)
            if word in self.out_of_context_words:
                out_of_context.append(word)

        return {
            "tokens": [token.text for token in doc],
            "avg_similarity": float(avg_similarity),
            "nmap_keywords_found": nmap_keywords_found,
            "out_of_context": out_of_context,
            "query_length": len(doc),
        }

    def understand_query(self, query: str) -> Dict:
        """Comprendre si la query est liee a NMAP"""

        analysis = self.analyze_query(query)

        avg_similarity = analysis["avg_similarity"]
        nmap_keywords_count = len(analysis["nmap_keywords_found"])
        out_of_context_count = len(analysis["out_of_context"])
        query_length = analysis["query_length"]

        # ============ LOGIQUE DE DECISION ============

        # Cas 1: Similarite tres elevee -> NMAP
        if avg_similarity > 0.50:
            is_relevant = True
            reason = f"Embedding similarity tres elevee ({avg_similarity:.2f})"

        # Cas 2: Similarite bonne + keywords NMAP -> NMAP
        elif avg_similarity > 0.40 and nmap_keywords_count >= 1:
            is_relevant = True
            reason = f"Bonne similarite ({avg_similarity:.2f}) + keywords NMAP"

        # Cas 3: Similarite faible + mots hors contexte -> NON-NMAP
        elif avg_similarity < 0.35 and out_of_context_count >= 2:
            is_relevant = False
            reason = f"Faible similarite ({avg_similarity:.2f}) + mots hors contexte"

        # Cas 4: Similarite moyenne + keywords NMAP -> NMAP
        elif avg_similarity > 0.30 and nmap_keywords_count >= 2:
            is_relevant = True
            reason = f"Similarite OK ({avg_similarity:.2f}) + multiple keywords NMAP"

        # Cas 5: Requete longue avec bonne similarite -> NMAP
        elif query_length > 15 and avg_similarity > 0.35:
            is_relevant = True
            reason = f"Requete longue ({query_length} mots) + similarite ({avg_similarity:.2f})"

        # Cas 6: Par defaut -> NON-NMAP
        else:
            is_relevant = False
            reason = f"Similarite insuffisante ({avg_similarity:.2f})"

        # ============ CONFIANCE ============

        if is_relevant:
            if avg_similarity > 0.55:
                confidence = "TRES_HAUTE"
            elif avg_similarity > 0.40:
                confidence = "MOYENNE_HAUTE"
            else:
                confidence = "MOYENNE"
        else:
            confidence = "BASSE"

        return {
            "query": query,
            "is_relevant": is_relevant,
            "decision": "ACCEPTE - NMAP" if is_relevant else "REJETE - Contexte non-NMAP",
            "analysis": {
                "tokens": analysis["tokens"],
                "nmap_keywords_found": analysis["nmap_keywords_found"],
                "out_of_context": analysis["out_of_context"],
                "query_length": analysis["query_length"],
            },
            "scores": {
                "embedding_similarity": round(avg_similarity, 3),
                "nmap_keywords_count": nmap_keywords_count,
                "out_of_context_count": out_of_context_count,
                "decision_reasoning": reason,
            },
            "confidence": confidence,
        }


# ============ TESTS ============

if __name__ == "__main__":
    print("=" * 100)
    print("AGENT NMAP AVEC SPACY EMBEDDINGS - TESTS")
    print("=" * 100)

    try:
        agent = NMAPEmbeddingAgent("nmap_domain.txt")

        test_cases = [
            ("scanner le port 80", True),
        ]

        print("\n")
        correct = 0
        for query, expected in test_cases:
            result = agent.understand_query(query)
            is_correct = result["is_relevant"] == expected
            correct += int(is_correct)

            status = "OK" if is_correct else "KO"
            print(f"{status} {result['decision']}")
            display_query = f'   Query: "{query[:60]}..."' if len(query) > 60 else f'   Query: "{query}"'
            print(display_query)
            print(f"   Similarity: {result['scores']['embedding_similarity']}")
            print(f"   Confiance: {result['confidence']}\n")

        print("=" * 100)
        print(f"Resultat: {correct}/{len(test_cases)} tests passes ({100*correct//len(test_cases)}%)")
        print("=" * 100)

    except Exception as e:
        print(f"Erreur: {e}")
        print("\nPour resoudre, executez:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
