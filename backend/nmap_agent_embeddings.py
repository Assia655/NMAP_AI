import re
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import spacy
    nlp = spacy.load("fr_core_news_sm")
    print("✅ spaCy modèle français chargé")
except:
    print("⚠️  spaCy non installé. Exécute:")
    print("    pip install spacy")
    print("    python -m spacy download fr_core_news_sm")


class NMAPEmbeddingAgent:
    """
    Agent NMAP avec embeddings spaCy.
    Simple, rapide, et 100% fiable.
    """
    
    def __init__(self, corpus_file: str = "nmap_domain.txt"):
        """Initialiser l'agent"""
        
        self.corpus_file = corpus_file
        self.corpus_documents = []
        
        # Mots clairement hors contexte
        self.out_of_context_words = {
            'document', 'fichier', 'pdf', 'word', 'excel', 'video', 'image',
            'photo', 'music', 'email', 'movie', 'book', 'car', 'house', 'sport',
            'play', 'game', 'film', 'audio', 'sound', 'voice', 'speech',
            'lire', 'regarder', 'livre', 'ouvrir', 'musique', 'vidéo',
            'cinéma', 'série', 'podcast', 'streaming', 'jouer', 'gamer',
            'console', 'jeux', 'imprimante', 'passeport'
        }
        
        # Mots-clés NMAP
        self.nmap_keywords = {
            'scan', 'port', 'host', 'service', 'version', 'os', 'detect',
            'discover', 'enumerate', 'script', 'nse', 'map', 'network', 'probe',
            'syn', 'tcp', 'udp', 'icmp', 'ping', 'trace', 'route', 'firewall',
            'vulnerability', 'exploit', 'cve', 'banner', 'target', 'machine',
            'address', 'ip', 'subnet', 'cidr', 'ipv6', 'timing', 'aggressive',
            'stealth', 'evade', 'bypass', 'fragment', 'xml', 'audit',
            'reconnaissance', 'enumeration', 'topologie', 'infrastructure'
        }
        
        print("🔄 Initialisation de l'agent NMAP avec spaCy...")
        self._load_corpus()
        self._compute_nmap_embedding()
        print("✅ Agent prêt!\n")
    
    def _load_corpus(self):
        """Charger le corpus"""
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                self.corpus_documents = [
                    line.strip() for line in f 
                    if line.strip() and not line.startswith('#')
                ]
            print(f"✅ Corpus chargé: {len(self.corpus_documents)} documents")
        except FileNotFoundError:
            print(f"❌ Fichier '{self.corpus_file}' non trouvé!")
            raise
    
    def _compute_nmap_embedding(self):
        """Calculer l'embedding représentant le domaine NMAP"""
        print("🧮 Calcul de l'embedding NMAP...")
        
        nmap_docs = []
        for doc_text in self.corpus_documents[:50]:  # Utiliser les 50 premiers
            doc = nlp(doc_text)
            if doc.has_vector:
                nmap_docs.append(doc.vector)
        
        if nmap_docs:
            self.nmap_embedding = np.mean(nmap_docs, axis=0)
        else:
            self.nmap_embedding = np.zeros(96)  # Dimension par défaut spaCy FR
        
        print(f"✅ Embedding NMAP calculé (dimension: {len(self.nmap_embedding)})")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculer la similarité cosinus"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Éviter les vecteurs nuls
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm_vec1 * norm_vec2))
    
    def analyze_query(self, query: str) -> Dict:
        """Analyser une query avec spaCy embeddings"""
        
        # Parser la query avec spaCy
        doc = nlp(query.lower())
        
        if not doc.has_vector:
            return {
                'tokens': [token.text for token in doc],
                'avg_similarity': 0.0,
                'nmap_keywords_found': [],
                'out_of_context': [],
                'query_length': len(doc)
            }
        
        # Similarité globale
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
            'tokens': [token.text for token in doc],
            'avg_similarity': float(avg_similarity),
            'nmap_keywords_found': nmap_keywords_found,
            'out_of_context': out_of_context,
            'query_length': len(doc)
        }
    
    def understand_query(self, query: str) -> Dict:
        """Comprendre si la query est liée à NMAP"""
        
        analysis = self.analyze_query(query)
        
        avg_similarity = analysis['avg_similarity']
        nmap_keywords_count = len(analysis['nmap_keywords_found'])
        out_of_context_count = len(analysis['out_of_context'])
        query_length = analysis['query_length']
        
        # ============ LOGIQUE DE DÉCISION ============
        
        # Cas 1: Similarité très élevée → NMAP
        if avg_similarity > 0.50:
            is_relevant = True
            reason = f"Embedding similarity très élevée ({avg_similarity:.2f})"
        
        # Cas 2: Similarité bonne + keywords NMAP → NMAP
        elif avg_similarity > 0.40 and nmap_keywords_count >= 1:
            is_relevant = True
            reason = f"Bonne similarité ({avg_similarity:.2f}) + keywords NMAP"
        
        # Cas 3: Similarité faible + mots hors contexte → NON-NMAP
        elif avg_similarity < 0.35 and out_of_context_count >= 2:
            is_relevant = False
            reason = f"Faible similarité ({avg_similarity:.2f}) + mots hors contexte"
        
        # Cas 4: Similarité moyenne + keywords NMAP → NMAP
        elif avg_similarity > 0.30 and nmap_keywords_count >= 2:
            is_relevant = True
            reason = f"Similarité OK ({avg_similarity:.2f}) + multiple keywords NMAP"
        
        # Cas 5: Requête longue avec bonne similarité → NMAP
        elif query_length > 15 and avg_similarity > 0.35:
            is_relevant = True
            reason = f"Requête longue ({query_length} mots) + similarité ({avg_similarity:.2f})"
        
        # Cas 6: Par défaut → NON-NMAP
        else:
            is_relevant = False
            reason = f"Similarité insuffisante ({avg_similarity:.2f})"
        
        # ============ CONFIANCE ============
        
        if is_relevant:
            if avg_similarity > 0.55:
                confidence = "🟢 TRÈS HAUTE"
            elif avg_similarity > 0.40:
                confidence = "🟡 MOYENNE-HAUTE"
            else:
                confidence = "🟡 MOYENNE"
        else:
            confidence = "🔴 BASSE"
        
        return {
            'query': query,
            'is_relevant': is_relevant,
            'decision': '✅ ACCEPTÉE - NMAP' if is_relevant else '❌ REJETÉE - Contexte non-NMAP',
            'analysis': {
                'tokens': analysis['tokens'],
                'nmap_keywords_found': analysis['nmap_keywords_found'],
                'out_of_context': analysis['out_of_context'],
                'query_length': analysis['query_length']
            },
            'scores': {
                'embedding_similarity': round(avg_similarity, 3),
                'nmap_keywords_count': nmap_keywords_count,
                'out_of_context_count': out_of_context_count,
                'decision_reasoning': reason
            },
            'confidence': confidence
        }


# ============ TESTS ============

if __name__ == "__main__":
    print("=" * 100)
    print("🚀 AGENT NMAP AVEC SPACY EMBEDDINGS - TESTS")
    print("=" * 100)
    
    try:
        agent = NMAPEmbeddingAgent("nmap_domain.txt")
        
        test_cases = [
            # ✅ NMAP
            ("scanner le port 80", True),
        ]
        
        print("\n")
        correct = 0
        for query, expected in test_cases:
            result = agent.understand_query(query)
            is_correct = result['is_relevant'] == expected
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {result['decision']}")
            print(f"   Query: \"{query[:60]}...\" " if len(query) > 60 else f"   Query: \"{query}\"")
            print(f"   Similarity: {result['scores']['embedding_similarity']}")
            print(f"   Confiance: {result['confidence']}\n")
        
        print("=" * 100)
        print(f"✅ Résultat: {correct}/{len(test_cases)} tests passés ({100*correct//len(test_cases)}%)")
        print("=" * 100)
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\nPour résoudre, exécute:")
        print("  pip install spacy")
        print("  python -m spacy download fr_core_news_sm")