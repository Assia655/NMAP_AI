# ==========================================
# rag_engine.py
# NMAP-AI RAG ENGINE - LOGIC LAYER
# Knowledge Graph RAG (Zero-Shot)
# ==========================================

from neo4j import GraphDatabase
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime


# ========== NEO4J CONNECTION ==========

class NeoConnection:
    """Gère la connexion à Neo4j"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.test_connection()
    
    def test_connection(self):
        """Teste la connexion"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) as count")
            count = result.single()['count']
            print(f"✅ Neo4j Connected - {count} nœuds trouvés")
    
    def close(self):
        self.driver.close()


# ========== KG RETRIEVAL - SCANS ==========

class ScanRetrieval:
    """Récupère les informations des scans du Knowledge Graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_scan_by_code(self, code: str) -> Dict:
        """Récupère les infos complètes d'un scan par son code"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scan {code: $code})
                RETURN s.code as code, s.name as name, s.description as description,
                       s.requires_root as requires_root, s.speed as speed, 
                       s.stealth as stealth, s.detection_capability as capability,
                       s.scan_type as scan_type, s.requires_open_port as requires_open,
                       s.requires_closed_port as requires_closed
                """,
                code=code
            )
            record = result.single()
            if record:
                return dict(record)
            return {}
    
    def get_scans_by_category(self, category: str) -> List[Dict]:
        """Récupère tous les scans d'une catégorie"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Category {name: $category})<-[:BELONGS_TO]-(s:Scan)
                RETURN s.code as code, s.name as name, s.description as description,
                       s.requires_root as requires_root, s.speed as speed, 
                       s.stealth as stealth, s.detection_capability as capability
                ORDER BY s.code
                """,
                category=category
            )
            return [dict(record) for record in result]
    
    def get_all_scans(self) -> List[Dict]:
        """Récupère TOUS les scans du graphe"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scan)
                RETURN s.code as code, s.name as name, s.description as description,
                       s.requires_root as requires_root, s.speed as speed, 
                       s.stealth as stealth, s.detection_capability as capability,
                       s.scan_type as scan_type
                ORDER BY s.code
                """
            )
            return [dict(record) for record in result]


# ========== KG RETRIEVAL - OPTIONS ==========

class OptionsRetrieval:
    """Récupère les informations des options du Knowledge Graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_option_by_code(self, code: str) -> Dict:
        """Récupère les infos d'une option"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (o:Option {code: $code})
                RETURN o.code as code, o.name as name, o.description as description,
                       o.category as category, o.requires_root as requires_root,
                       o.works_without_scan as works_without_scan
                """,
                code=code
            )
            record = result.single()
            if record:
                return dict(record)
            return {}
    
    def get_options_by_category(self, category: str) -> List[Dict]:
        """Récupère toutes les options d'une catégorie"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (o:Option {category: $category})
                RETURN o.code as code, o.name as name, o.description as description,
                       o.speed_impact as speed_impact, o.stealth_impact as stealth_impact
                ORDER BY o.code
                """,
                category=category
            )
            return [dict(record) for record in result]


# ========== KG RETRIEVAL - RELATIONS ==========

class RelationsRetrieval:
    """Récupère les relations entre nœuds du Knowledge Graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_compatible_options(self, scan_code: str) -> List[Dict]:
        """Récupère les options compatibles avec un scan"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scan {code: $code})<-[:WORKS_WITH]-(o:Option)
                RETURN o.code as code, o.name as name, o.description as description,
                       o.category as category
                ORDER BY o.code
                """,
                code=scan_code
            )
            return [dict(record) for record in result]
    
    def get_conflicting_scans(self, scan_code: str) -> List[Dict]:
        """Récupère les scans qui entrent en CONFLIT"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s1:Scan {code: $code})-[r:CONFLICTS_WITH]->(s2:Scan)
                RETURN s2.code as code, s2.name as name, r.reason as reason
                """,
                code=scan_code
            )
            return [dict(record) for record in result]
    
    def get_compatible_scans(self, scan_code: str) -> List[Dict]:
        """Récupère les scans COMPATIBLES"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s1:Scan {code: $code})-[r:COMPATIBLE_WITH]->(s2:Scan)
                RETURN s2.code as code, s2.name as name, r.note as note
                """,
                code=scan_code
            )
            return [dict(record) for record in result]
    
    def get_scan_category(self, scan_code: str) -> Dict:
        """Récupère la catégorie d'un scan"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scan {code: $code})-[:BELONGS_TO]->(c:Category)
                RETURN c.name as name, c.description as description
                """,
                code=scan_code
            )
            record = result.single()
            if record:
                return dict(record)
            return {}


# ========== KG RETRIEVAL - EXAMPLES ==========

class ExamplesRetrieval:
    """Récupère les exemples du Knowledge Graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_similar_examples(self, intent: str) -> List[Dict]:
        """Récupère des exemples similaires basés sur l'intent"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Example)
                WHERE e.use_case CONTAINS $intent OR e.difficulty = $intent
                RETURN e.name as name, e.command as command, 
                       e.description as description, e.use_case as use_case
                LIMIT 3
                """,
                intent=intent
            )
            return [dict(record) for record in result]


# ========== KG RETRIEVAL - DEPENDENCIES ==========

class DependencyRetrieval:
    """Récupère les dépendances et privilèges"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_scan_dependencies(self, scan_code: str) -> Dict:
        """Récupère les dépendances d'un scan"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scan {code: $code})
                RETURN s.requires_open_port as requires_open,
                       s.requires_closed_port as requires_closed,
                       s.dependency as dependency_msg
                """,
                code=scan_code
            )
            record = result.single()
            if record:
                return {
                    'requires_open_port': record['requires_open'],
                    'requires_closed_port': record['requires_closed'],
                    'dependency_message': record['dependency_msg']
                }
            return {}
    
    def get_privilege_requirements(self, scan_code: str) -> Dict:
        """Récupère les exigences de privilèges"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Privilege)-[:REQUIRED_FOR]->(s:Scan {code: $code})
                RETURN p.level as level, p.description as description
                """,
                code=scan_code
            )
            record = result.single()
            if record:
                return dict(record)
            return {}
    
    def get_validation_warnings(self, scan_code: str) -> List[Dict]:
        """Récupère les warnings de validation"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (v:Validation)-[:VALIDATES]->(s:Scan {code: $code})
                RETURN v.name as name, v.rule as rule, v.severity as severity
                """,
                code=scan_code
            )
            return [dict(record) for record in result]


# ========== RAG QUERY ENGINE ==========

class RAGQueryEngine:
    """Moteur RAG principal - RETRIEVAL"""
    
    def __init__(self, driver):
        self.driver = driver
        self.scans = ScanRetrieval(driver)
        self.options = OptionsRetrieval(driver)
        self.relations = RelationsRetrieval(driver)
        self.dependencies = DependencyRetrieval(driver)
        self.examples = ExamplesRetrieval(driver)
    
    def query_full_context(self, scan_code: str) -> Dict:
        """Récupère TOUT le contexte d'un scan depuis le KG"""
        scan_info = self.scans.get_scan_by_code(scan_code)
        if not scan_info:
            return {}
        
        return {
            'scan': scan_info,
            'compatible_options': self.relations.get_compatible_options(scan_code),
            'conflicting_scans': self.relations.get_conflicting_scans(scan_code),
            'compatible_scans': self.relations.get_compatible_scans(scan_code),
            'category': self.relations.get_scan_category(scan_code),
            'dependencies': self.dependencies.get_scan_dependencies(scan_code),
            'privilege_info': self.dependencies.get_privilege_requirements(scan_code),
            'validation_warnings': self.dependencies.get_validation_warnings(scan_code)
        }
    
    def query_by_intent(self, intent: str) -> Dict:
        """Requête RAG par INTENT"""
        intent_map = {
            'host_discovery': 'Host Discovery',
            'service_detection': 'Version Detection',
            'os_detection': 'OS Detection',
            'firewall_detection': 'Firewall/IDS Evasion',
            'stealth_scan': 'Port Scanning',
            'port_scanning': 'Port Scanning',
            'general_scan': 'Port Scanning',
            'vuln_scan': 'Advanced Scanning',
            'fast_scan': 'Port Scanning'
        }
        
        category = intent_map.get(intent, 'Port Scanning')
        scans = self.scans.get_scans_by_category(category)
        examples = self.examples.get_similar_examples(intent)
        
        return {
            'intent': intent,
            'mapped_category': category,
            'recommended_scans': scans,
            'examples': examples,
            'total_options': len(scans)
        }


# ========== INTENT CLASSIFIER ==========

class IntentClassifier:
    """Classifie l'intention de l'utilisateur - Zero-shot"""
    
    def __init__(self):
        self.intent_patterns = {
            'host_discovery': ['ping', 'host', 'alive', 'discovery', 'detect host', 'check if', 'up', 'online'],
            'port_scanning': ['port', 'scan port', 'open port', 'check port', 'which ports'],
            'service_detection': ['service', 'version', 'detect service', 'banner', 'what service', 'running'],
            'os_detection': ['os', 'operating system', 'fingerprint', 'detect os', 'which os', 'system'],
            'stealth_scan': ['stealth', 'hidden', 'evade', 'stealthy', 'quiet', 'undetected', 'covert'],
            'fast_scan': ['fast', 'quick', 'rapid', 'speed', 'quickly'],
            'firewall_detection': ['firewall', 'ids', 'ips', 'detection', 'filter'],
            'vuln_scan': ['vuln', 'vulnerability', 'exploit', 'cve', 'script', 'aggressive']
        }
    
    def classify(self, user_query: str) -> Tuple[str, float]:
        """Classifie l'intention utilisateur"""
        query_lower = user_query.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
            
            if len(patterns) > 0:
                intent_scores[intent] = score / len(patterns)
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return ('general_scan', 0.5)
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        return (best_intent, confidence)
    
    def extract_target(self, user_query: str) -> Optional[str]:
        """Extrait la cible (IP/hostname)"""
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?\b'
        ip_match = re.search(ip_pattern, user_query)
        if ip_match:
            return ip_match.group(0)
        
        hostname_pattern = r'\b([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}\b'
        hostname_match = re.search(hostname_pattern, user_query.lower())
        if hostname_match:
            return hostname_match.group(0)
        
        return None
    
    def extract_ports(self, user_query: str) -> Optional[str]:
        """Extrait les ports"""
        port_pattern = r'ports?\s+([0-9,\-]+)'
        match = re.search(port_pattern, user_query.lower())
        if match:
            return match.group(1)
        
        return None


# ========== COMMAND GENERATOR ==========

class CommandGenerator:
    """Génère des commandes Nmap VALIDES en Zero-Shot avec RAG"""
    
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag = rag_engine
    
    def generate_from_intent(self, intent: str, target: str, 
                            ports: Optional[str] = None,
                            stealth: bool = False,
                            fast: bool = False) -> Dict:
        """Génère une commande Nmap à partir de l'intent avec contexte KG complet"""
        
        # ÉTAPE 1: Récupérer les scans recommandés pour cet intent
        rag_result = self.rag.query_by_intent(intent)
        recommended_scans = rag_result['recommended_scans']
        
        if not recommended_scans:
            return {
                'command': None,
                'explanation': 'No scans found for this intent',
                'warnings': ['Intent not recognized'],
                'kg_context': {}
            }
        
        # ÉTAPE 2: Sélectionner le meilleur scan
        primary_scan = self._select_best_scan(recommended_scans, stealth, fast)
        
        # ÉTAPE 3: Récupérer TOUT le contexte du scan depuis le KG
        kg_context = self.rag.query_full_context(primary_scan['code'])
        
        # ÉTAPE 4: Construire la commande avec le contexte KG
        command_parts = ['nmap']
        warnings = []
        explanation_parts = []
        used_options = []
        
        # Scan principal
        command_parts.append(primary_scan['code'])
        explanation_parts.append(f"Using {primary_scan['name']}: {primary_scan['description']}")
        
        # Privilèges requis (depuis KG)
        if kg_context.get('privilege_info'):
            priv = kg_context['privilege_info']
            warnings.append(f"⚠️ Requires {priv['level']} privileges: {priv['description']}")
        
        # Options compatibles selon intent (depuis KG)
        compatible_opts = kg_context.get('compatible_options', [])
        
        if intent == 'service_detection':
            # Ajouter -sV si compatible
            sv_opt = next((o for o in compatible_opts if o['code'] == '-sV'), None)
            if sv_opt:
                command_parts.append('-sV')
                explanation_parts.append(f"Version detection: {sv_opt['description']}")
                used_options.append(sv_opt)
        
        if intent == 'os_detection':
            # Ajouter -O si compatible
            o_opt = next((o for o in compatible_opts if o['code'] == '-O'), None)
            if o_opt:
                command_parts.append('-O')
                explanation_parts.append(f"OS detection: {o_opt['description']}")
                used_options.append(o_opt)
                warnings.append("⚠️ -O requires root and both open and closed ports")
        
        # Timing selon fast/stealth
        if fast:
            t4_opt = next((o for o in compatible_opts if o['code'] == '-T4'), None)
            if t4_opt:
                command_parts.append('-T4')
                explanation_parts.append("Fast timing (T4)")
        elif stealth:
            t1_opt = next((o for o in compatible_opts if o['code'] == '-T1'), None)
            if t1_opt:
                command_parts.append('-T1')
                explanation_parts.append("Stealthy timing (T1)")
        
        # Ports
        if ports:
            command_parts.append(f'-p {ports}')
            explanation_parts.append(f"Scanning ports: {ports}")
        
        # Target
        command_parts.append(target)
        
        # Validation warnings depuis KG
        val_warnings = kg_context.get('validation_warnings', [])
        for vw in val_warnings:
            warnings.append(f"⚠️ {vw['name']}: {vw['rule']}")
        
        # Conflits depuis KG
        conflicts = kg_context.get('conflicting_scans', [])
        if conflicts:
            conflict_codes = [c['code'] for c in conflicts]
            warnings.append(f"⚠️ Conflicts with: {', '.join(conflict_codes)}")
        
        final_command = ' '.join(command_parts)
        
        return {
            'command': final_command,
            'explanation': ' | '.join(explanation_parts),
            'warnings': warnings,
            'scan_info': primary_scan,
            'requires_root': primary_scan.get('requires_root', False),
            'kg_context': kg_context,
            'used_options': used_options,
            'examples': rag_result.get('examples', [])
        }
    
    def _select_best_scan(self, scans: List[Dict], stealth: bool, fast: bool) -> Dict:
        """Sélectionne le meilleur scan selon les critères"""
        if stealth:
            stealth_scans = [s for s in scans if s.get('stealth') in ['high', 'very_high']]
            if stealth_scans:
                return stealth_scans[0]
        
        if fast:
            fast_scans = [s for s in scans if s.get('speed') in ['fast', 'very_fast']]
            if fast_scans:
                return fast_scans[0]
        
        return scans[0] if scans else {}


# ========== COMMAND VALIDATOR ==========

class CommandValidator:
    """Valide les commandes Nmap générées avec le KG"""
    
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag = rag_engine
    
    def validate(self, command: str) -> Dict:
        """Valide une commande Nmap avec contexte KG"""
        errors = []
        warnings = []
        score = 1.0
        
        parts = command.split()
        
        if not parts or parts[0] != 'nmap':
            errors.append("Command must start with 'nmap'")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'score': 0.0}
        
        scan_codes = [p for p in parts if p.startswith('-s')]
        
        # Vérifier conflits via KG
        for scan1 in scan_codes:
            scan_info = self.rag.scans.get_scan_by_code(scan1)
            if not scan_info:
                errors.append(f"Unknown scan: {scan1}")
                score -= 0.3
                continue
            
            # Récupérer conflits depuis KG
            conflicts = self.rag.relations.get_conflicting_scans(scan1)
            for conflict in conflicts:
                if conflict['code'] in scan_codes:
                    errors.append(f"Conflict: {scan1} ↔ {conflict['code']} - {conflict['reason']}")
                    score -= 0.5
            
            # Vérifier privilèges via KG
            priv_info = self.rag.dependencies.get_privilege_requirements(scan1)
            if priv_info and priv_info.get('level') == 'root':
                warnings.append(f"{scan1} requires root: {priv_info['description']}")
        
        # Vérifier target
        if not any(self._is_target(p) for p in parts):
            errors.append("No target specified (IP or hostname)")
            score -= 0.4
        
        score = max(0.0, score)
        valid = len(errors) == 0 and score >= 0.5
        
        return {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'score': score
        }
    
    def _is_target(self, part: str) -> bool:
        """Vérifie si c'est une cible"""
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(/\d{1,2})?$'
        if re.match(ip_pattern, part):
            return True
        
        hostname_pattern = r'^([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}$'
        if re.match(hostname_pattern, part.lower()):
            return True
        
        return False


# ========== RAG PIPELINE COMPLETE ==========

class NmapRAGPipeline:
    """Pipeline RAG Complet (Easy Tier) - Utilise TOUT le Knowledge Graph"""
    
    def __init__(self, driver):
        self.rag = RAGQueryEngine(driver)
        self.intent_classifier = IntentClassifier()
        self.generator = CommandGenerator(self.rag)
        self.validator = CommandValidator(self.rag)
    
    def process_query(self, user_query: str) -> Dict:
        """
        Pipeline complet: NL Query → Nmap Command
        Utilise TOUT le Knowledge Graph pour la génération
        """
        
        # ÉTAPE 1: Classify Intent
        intent, confidence = self.intent_classifier.classify(user_query)
        
        # ÉTAPE 2: Extract Parameters
        target = self.intent_classifier.extract_target(user_query)
        ports = self.intent_classifier.extract_ports(user_query)
        
        stealth = 'stealth' in user_query.lower() or 'hidden' in user_query.lower()
        fast = 'fast' in user_query.lower() or 'quick' in user_query.lower()
        
        if not target:
            return {
                'success': False,
                'error': 'No target specified. Please provide an IP address or hostname.',
                'query': user_query,
                'suggestion': 'Example: "Scan 192.168.1.1 for open ports"'
            }
        
        # ÉTAPE 3: Generate Command (RAG with full KG context)
        generation_result = self.generator.generate_from_intent(
            intent=intent,
            target=target,
            ports=ports,
            stealth=stealth,
            fast=fast
        )
        
        if not generation_result['command']:
            return {
                'success': False,
                'error': 'Failed to generate command',
                'query': user_query,
                'intent': intent
            }
        
        # ÉTAPE 4: Validate Command (with KG)
        validation_result = self.validator.validate(generation_result['command'])
        
        # ÉTAPE 5: Build Final Result with KG Context
        result = {
            'success': validation_result['valid'],
            'query': user_query,
            'intent': intent,
            'confidence': confidence,
            'target': target,
            'ports': ports,
            'command': generation_result['command'],
            'explanation': generation_result['explanation'],
            'scan_info': generation_result['scan_info'],
            'requires_root': generation_result['requires_root'],
            'validation': validation_result,
            'warnings': generation_result['warnings'] + validation_result['warnings'],
            'kg_context': {
                'category': generation_result['kg_context'].get('category', {}),
                'compatible_scans': generation_result['kg_context'].get('compatible_scans', []),
                'used_options': generation_result.get('used_options', []),
                'examples': generation_result.get('examples', [])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result