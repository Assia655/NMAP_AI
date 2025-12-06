from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Load domain corpus
# --------------------------
with open("nmap_domain.txt", "r", encoding="utf-8") as f:
    domain_lines = [l.strip() for l in f.readlines() if l.strip()]

# --------------------------
# TF-IDF with stopwords
# --------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
domain_vectors = vectorizer.fit_transform(domain_lines)

DOMAIN_TOKENS = set(vectorizer.get_feature_names_out())

# --------------------------
# Compute cosine similarity
# --------------------------
def compute_similarity(query: str) -> float:
    vec = vectorizer.transform([query])
    sims = cosine_similarity(vec, domain_vectors)[0]
    return float(max(sims))


# --------------------------
# Count domain tokens
# --------------------------
def count_domain_tokens(text):
    words = text.lower().split()
    return sum(1 for w in words if w in DOMAIN_TOKENS)


# --------------------------
# Intent analysis
# --------------------------
def analyze_intent(text: str) -> str:
    t = text.lower()
    if "scan" in t or "port" in t:
        return "Analyse de scan de ports."
    if "version" in t or "-sv" in t:
        return "Détection de version."
    if "os" in t or "-o" in t:
        return "Détection du système d'exploitation."
    if "udp" in t:
        return "Scan UDP."
    if "script" in t or "nse" in t:
        return "Utilisation de scripts NSE."
    return "Requête liée à Nmap détectée."


# --------------------------
# Main Agent Logic
# --------------------------
THRESHOLD = 0.50
MIN_WORDS = 4
MIN_DOMAIN_TOKENS = 2
MIN_DOMAIN_RATIO = 0.60


def comprehension_agent(user_message: str) -> dict:
    text = user_message.lower()
    words = text.split()

    # 1) Reject short sentences
    if len(words) < MIN_WORDS:
        return {
            "is_in_context": False,
            "message": "Votre demande ne concerne pas Nmap.",
            "confidence": 0.0
        }

    # 2) Count domain tokens
    domain_count = count_domain_tokens(text)
    if domain_count < MIN_DOMAIN_TOKENS:
        return {
            "is_in_context": False,
            "message": "Votre demande ne concerne pas Nmap.",
            "confidence": 0.0
        }

    # 3) Domain token ratio rule
    domain_ratio = domain_count / len(words)
    if domain_ratio < MIN_DOMAIN_RATIO:
        return {
            "is_in_context": False,
            "message": "Votre demande ne concerne pas Nmap.",
            "confidence": 0.0
        }

    # 4) TF-IDF Similarity threshold
    similarity = compute_similarity(user_message)
    
    if similarity < THRESHOLD:
        return {
            "is_in_context": False,
            "message": "Votre demande ne concerne pas Nmap.",
            "confidence": similarity
        }

    # 5) Valid Nmap query
    return {
        "is_in_context": True,
        "analysis": analyze_intent(user_message),
        "raw_intent": user_message,
        "confidence": similarity
    }
