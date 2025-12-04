from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le corpus
with open("nmap_domain.txt", "r", encoding="utf-8") as f:
    domain_lines = [l.strip() for l in f.readlines() if l.strip()]

# TF-IDF Vectorisation
vectorizer = TfidfVectorizer()
domain_vectors = vectorizer.fit_transform(domain_lines)


# --- Embedding de la requête ---
def embed_query(query):
    return vectorizer.transform([query])


# --- Similarité cosinus ---
def compute_similarity(query):
    q_vec = embed_query(query)
    sims = cosine_similarity(q_vec, domain_vectors)[0]
    return float(max(sims))


# --- Analyse d’intention simple ---
def analyze_intent(text):
    t = text.lower()

    if "port" in t or "scan" in t:
        return "Analyse de scan de ports."
    if "version" in t or "-sv" in t:
        return "Détection de version."
    if "os" in t or "-o" in t:
        return "Détection de système d'exploitation."
    if "udp" in t:
        return "Scan UDP."
    if "script" in t or "nse" in t:
        return "Utilisation de scripts NSE."

    return "Requête liée à Nmap détectée."


# --- Agent principal ---
THRESHOLD = 0.20

def comprehension_agent(user_message):
    similarity = compute_similarity(user_message)

    if similarity < THRESHOLD:
        return {
            "is_in_context": False,
            "message": "Votre demande ne concerne pas Nmap. Exemple: 'Scanner les ports ouverts sur 192.168.1.1'",
            "confidence": similarity
        }

    return {
        "is_in_context": True,
        "analysis": analyze_intent(user_message),
        "raw_intent": user_message,
        "confidence": similarity
    }

