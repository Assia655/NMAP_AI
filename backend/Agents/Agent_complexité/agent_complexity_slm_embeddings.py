"""
Agent de Complexité NMAP - SLM + Embeddings
============================================

Architecture hybride:
1. Embeddings (Sentence-BERT) pour vectorisation sémantique
2. KNN pour classification initiale
3. SLM (T5-small) pour validation et génération de raisonnement

Dépendances:
    pip install transformers torch sentence-transformers scikit-learn

Usage:
    from agent_complexity_slm_embeddings import ComplexityClassifierSLM
    
    classifier = ComplexityClassifierSLM()
    classifier.train()
    result = classifier.classify("Scan all ports using SYN scan")
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ComplexityClassifierSLM:
    """Classifier hybride avec SLM + Embeddings"""
    
    def __init__(self, dataset_path="nmap_complexity_dataset.txt"):
        """
        Initialise le classifier hybride
        
        Args:
            dataset_path (str): Chemin vers le dataset
        """
        self.dataset_path = Path(__file__).parent / dataset_path
        
        # Modèles
        self.slm_model = None
        self.slm_tokenizer = None
        self.embedding_model = None
        self.knn_classifier = None
        
        # Configuration
        self.classes = ['easy', 'medium', 'hard']
        self.model_map = {
            'easy': 'KG-RAG (Knowledge Graph)',
            'medium': 'LoRA fine-tuned (T5-small / Phi-4)',
            'hard': 'Diffusion-based synthesis'
        }
        
        # Données d'entraînement
        self.train_embeddings = None
        self.train_labels = None
        self.train_queries = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    def load_dataset(self):
        """Charge le dataset annoté"""
        queries = []
        labels = []
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if '|' in line:
                    query, label = line.rsplit('|', 1)
                    query = query.strip()
                    label = label.strip().lower()
                    
                    if label in self.classes:
                        queries.append(query)
                        labels.append(label)
        
        print(f"✓ Dataset chargé: {len(queries)} exemples")
        print(f"  - EASY: {labels.count('easy')}")
        print(f"  - MEDIUM: {labels.count('medium')}")
        print(f"  - HARD: {labels.count('hard')}")
        
        return queries, labels
    
    
    def train(self):
        """Entraîne le classifier hybride"""
        print("\n" + "="*70)
        print("  ENTRAÎNEMENT - SLM + EMBEDDINGS")
        print("="*70)
        
        # ==============================
        # 1. CHARGER LE SLM (T5-small)
        # ==============================
        print("\n⏳ [1/4] Chargement du SLM (T5-small)...")
        try:
            self.slm_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.slm_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            self.slm_model.to(self.device)
            self.slm_model.eval()
            print(f"✓ SLM chargé sur {self.device}")
        except Exception as e:
            print(f"⚠️ Erreur SLM: {e}")
            print("Utilisation de google/flan-t5-small comme alternative...")
            self.slm_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
            self.slm_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
            self.slm_model.to(self.device)
            self.slm_model.eval()
            print(f"✓ FLAN-T5 chargé sur {self.device}")
        
        # ==============================
        # 2. CHARGER EMBEDDINGS
        # ==============================
        print("\n⏳ [2/4] Chargement du modèle d'embeddings (Sentence-BERT)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embeddings chargés")
        
        # ==============================
        # 3. PRÉPARER LES DONNÉES
        # ==============================
        print("\n⏳ [3/4] Préparation des données...")
        queries, labels = self.load_dataset()
        self.train_queries = queries
        
        # Générer les embeddings
        print("   → Génération des embeddings...")
        embeddings = self.embedding_model.encode(
            queries, 
            show_progress_bar=True,
            batch_size=32
        )
        print(f"   ✓ Embeddings générés: shape {embeddings.shape}")
        
        self.train_embeddings = embeddings
        self.train_labels = np.array(labels)
        
        # ==============================
        # 4. ENTRAÎNER KNN
        # ==============================
        print("\n⏳ [4/4] Entraînement du classifier KNN...")
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=7,
            metric='cosine',
            weights='distance'
        )
        self.knn_classifier.fit(embeddings, labels)
        
        accuracy = self.knn_classifier.score(embeddings, labels)
        print(f"✓ KNN entraîné - Accuracy: {accuracy*100:.2f}%")
        
        print("\n" + "="*70)
        print("  ✓ ENTRAÎNEMENT TERMINÉ")
        print("="*70)
        print(f"  🤖 SLM: T5-small")
        print(f"  🧠 Embeddings: all-MiniLM-L6-v2 (dim={embeddings.shape[1]})")
        print(f"  📊 Accuracy: {accuracy*100:.2f}%")
        print(f"  💻 Device: {self.device}")
        print("="*70 + "\n")
        
        return {
            'accuracy': accuracy,
            'method': 'SLM (T5-small) + Embeddings (Sentence-BERT)',
            'embedding_dim': embeddings.shape[1],
            'n_samples': len(queries),
            'device': str(self.device)
        }
    
    
    def _generate_slm_reasoning(self, query, knn_level, probabilities, similar_queries):
        """
        Génère un raisonnement avec le SLM
        
        Le SLM analyse la requête et valide/ajuste la décision du KNN
        """
        # Construire un prompt plus structuré pour T5
        prompt = f"""Explain why this nmap query is {knn_level} complexity:
Query: {query[:120]}
Reason:"""
        
        # Tokenizer
        inputs = self.slm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=200,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Générer avec le SLM
        with torch.no_grad():
            outputs = self.slm_model.generate(
                inputs.input_ids,
                max_length=60,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Décoder la réponse
        slm_response = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Construire le raisonnement enrichi
        reasoning_parts = []
        
        # Probabilités KNN
        prob_str = " | ".join([
            f"{cls.upper()}: {prob:.1%}"
            for cls, prob in probabilities.items()
        ])
        reasoning_parts.append(f"KNN: {prob_str}")
        
        # Similarité
        if similar_queries:
            top_sim = similar_queries[0]['similarity']
            reasoning_parts.append(f"Sim: {top_sim:.1%}")
        
        # Validation SLM (plus concise)
        if slm_response and len(slm_response) > 5:
            # Limiter à 50 caractères pour éviter les réponses trop longues
            slm_short = slm_response[:50] + "..." if len(slm_response) > 50 else slm_response
            reasoning_parts.append(f"SLM: {slm_short}")
        
        return " — ".join(reasoning_parts)
    
    
    def _validate_with_slm(self, query, knn_level, probabilities):
        """
        Valide la décision KNN avec le SLM
        
        Le SLM intervient UNIQUEMENT si KNN est incertain
        
        Returns:
            str: Niveau final validé par le SLM
        """
        # Trouver la prédiction KNN avec la plus haute probabilité
        best_level = max(probabilities, key=probabilities.get)
        best_prob = probabilities[best_level]
        
        # Si le KNN est confiant (>65%), on GARDE TOUJOURS sa décision
        if best_prob > 0.65:
            return best_level
        
        # Sinon (cas ambigu), on consulte le SLM
        prompt = f"""Classify this nmap query complexity level.
Query: {query[:200]}
Options: easy, medium, hard
Answer with only one word:"""
        
        inputs = self.slm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.slm_model.generate(
                inputs.input_ids,
                max_length=5,  # Seulement 1 mot
                num_beams=3,
                do_sample=False,
                early_stopping=True
            )
        
        slm_prediction = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
        
        # Extraire le niveau du texte généré
        for level in self.classes:
            if level in slm_prediction:
                return level
        
        # Si pas de correspondance, garder la prédiction KNN
        return best_level
    
    
    def classify(self, query):
        """
        Classifie une requête avec SLM + Embeddings
        
        Pipeline:
        1. Génération d'embedding (Sentence-BERT)
        2. Classification KNN
        3. Validation avec SLM (T5)
        4. Génération de raisonnement par SLM
        
        Args:
            query (str): Requête à classifier
            
        Returns:
            dict: Résultat de la classification
        """
        if self.knn_classifier is None or self.slm_model is None:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        # ==============================
        # ÉTAPE 1: EMBEDDING
        # ==============================
        query_embedding = self.embedding_model.encode([query])
        
        # ==============================
        # ÉTAPE 2: CLASSIFICATION KNN
        # ==============================
        knn_level = str(self.knn_classifier.predict(query_embedding)[0])  # Convertir en str
        knn_proba = self.knn_classifier.predict_proba(query_embedding)[0]
        
        # Probabilités KNN
        proba_dict = {}
        for i, cls in enumerate(self.knn_classifier.classes_):
            proba_dict[str(cls)] = float(knn_proba[i])  # Convertir en str Python standard
        
        # ==============================
        # ÉTAPE 3: VALIDATION SLM
        # ==============================
        # Déterminer si on doit utiliser le SLM
        best_level = max(proba_dict, key=proba_dict.get)
        best_prob = proba_dict[best_level]
        slm_was_used = best_prob <= 0.65  # SLM utilisé si KNN incertain
        
        if slm_was_used:
            final_level = self._validate_with_slm(query, knn_level, proba_dict)
        else:
            final_level = best_level
        
        final_confidence = float(proba_dict[final_level])
        
        # ==============================
        # ÉTAPE 4: SIMILARITÉ
        # ==============================
        similarities = cosine_similarity(query_embedding, self.train_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        similar_queries = [
            {
                'query': self.train_queries[i],
                'label': str(self.train_labels[i]),  # Convertir en str
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]
        
        # ==============================
        # ÉTAPE 5: RAISONNEMENT SLM
        # ==============================
        reasoning = self._generate_slm_reasoning(
            query, 
            knn_level, 
            proba_dict, 
            similar_queries
        )
        
        return {
            'query': query,
            'level': final_level,
            'confidence': round(final_confidence, 2),
            'probabilities': {k: round(v, 3) for k, v in proba_dict.items()},
            'reasoning': reasoning,
            'recommended_model': self.model_map[final_level],
            'method': 'SLM (T5-small) + Embeddings (Sentence-BERT)',
            'similar_queries': similar_queries,
            'slm_validation': {
                'knn_prediction': knn_level,
                'slm_prediction': final_level,
                'validated': final_level == knn_level,
                'slm_was_consulted': slm_was_used,  # Nouveau : indique si SLM a été utilisé
                'knn_confidence': round(best_prob, 2)  # Confiance du KNN
            }
        }
    
    
    def batch_classify(self, queries):
        """Classifie plusieurs requêtes"""
        return [self.classify(q) for q in queries]


# ==============================================
# TEST LOCAL
# ==============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TEST - Agent Complexité SLM + EMBEDDINGS")
    print("="*70)
    
    # Créer et entraîner
    classifier = ComplexityClassifierSLM()
    metrics = classifier.train()
    
    # Tests
    test_queries = [
        "nmap -sS 192.168.1.1",
        "Scan all ports with version detection",
        "Create Python script to automate nmap with custom NSE and bypass firewall"
    ]
    
    print("\n" + "="*70)
    print("  TESTS DE CLASSIFICATION")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/3")
        print(f"{'='*70}")
        
        result = classifier.classify(query)
        
        print(f"\n📝 Query: '{query}'")
        print(f"\n🎯 Résultat:")
        print(f"   → Level: {result['level'].upper()}")
        print(f"   → Confidence: {result['confidence']*100:.1f}%")
        print(f"   → Probabilities: {result['probabilities']}")
        
        print(f"\n🤖 SLM Validation:")
        print(f"   → KNN prédit: {result['slm_validation']['knn_prediction'].upper()}")
        print(f"   → SLM valide: {result['slm_validation']['slm_prediction'].upper()}")
        print(f"   → Accord: {'✅' if result['slm_validation']['validated'] else '⚠️'}")
        
        print(f"\n💭 Raisonnement:")
        print(f"   {result['reasoning']}")
        
        print(f"\n🔍 Requête similaire:")
        print(f"   → {result['similar_queries'][0]['query']}")
        print(f"   → Label: {result['similar_queries'][0]['label'].upper()}")
        print(f"   → Similarité: {result['similar_queries'][0]['similarity']:.1%}")
    
    print("\n" + "="*70)
    print("  ✓ TESTS TERMINÉS")
    print("="*70 + "\n")