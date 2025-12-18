// ==========================================
// NMAP-AI COMPLETE KNOWLEDGE GRAPH
// Neo4j Cypher - ÉTAPE 1: BUILD KG
// ==========================================

// ========== SECTION 1: CATÉGORIES ==========
y {name: "OS Detection", description: "Détection du système d'exploitation"})
CREATE (cat_version:Category {name: "CREATE (cat_discovery:Category {name: "Host Discovery", description: "Découverte des hôtes actifs"})
CREATE (cat_port_scan:Category {name: "Port Scanning", description: "Scan des ports TCP/UDP"})
CREATE (cat_os_detection:CategorVersion Detection", description: "Détection des versions de services"})
CREATE (cat_firewall:Category {name: "Firewall/IDS Evasion", description: "Techniques d'évasion des pare-feu"})
CREATE (cat_advanced:Category {name: "Advanced Scanning", description: "Scans avancés et personnalisés"})
CREATE (cat_output:Category {name: "Output", description: "Formats de sortie des résultats"})
CREATE (cat_privilege:Category {name: "Privilege Requirements", description: "Exigences de privilèges"});

// ========== SECTION 2: PORT RANGES (NOUVEAUX NŒUDS) ==========
CREATE (port_http:PortRange {name: "80", service: "HTTP", protocol: "TCP", severity: "low"})
CREATE (port_https:PortRange {name: "443", service: "HTTPS", protocol: "TCP", severity: "low"})
CREATE (port_ssh:PortRange {name: "22", service: "SSH", protocol: "TCP", severity: "medium"})
CREATE (port_rdp:PortRange {name: "3389", service: "RDP", protocol: "TCP", severity: "high"})
CREATE (port_dns:PortRange {name: "53", service: "DNS", protocol: "UDP", severity: "medium"})
CREATE (port_snmp:PortRange {name: "161", service: "SNMP", protocol: "UDP", severity: "high"})
CREATE (port_smb:PortRange {name: "445", service: "SMB", protocol: "TCP", severity: "critical"})
CREATE (port_telnet:PortRange {name: "23", service: "TELNET", protocol: "TCP", severity: "critical"})
CREATE (port_ftp:PortRange {name: "21", service: "FTP", protocol: "TCP", severity: "high"})
CREATE (port_smtp:PortRange {name: "25", service: "SMTP", protocol: "TCP", severity: "medium"})
CREATE (port_mysql:PortRange {name: "3306", service: "MYSQL", protocol: "TCP", severity: "high"})
CREATE (port_rdatabase:PortRange {name: "5432", service: "POSTGRESQL", protocol: "TCP", severity: "high"})
CREATE (port_range_top100:PortRange {name: "1-100", service: "Top 100 ports", protocol: "TCP", severity: "varies"})
CREATE (port_range_all:PortRange {name: "1-65535", service: "All ports", protocol: "TCP", severity: "varies"})
CREATE (port_range_common:PortRange {name: "80,443,22,3389,445", service: "Most common", protocol: "TCP", severity: "high"});

// ========== SECTION 3: PRIVILEGE REQUIREMENTS (CENTRALISER) ==========
CREATE (priv_root:Privilege {level: "root", description: "Requires root/sudo access", priority: "high"})
CREATE (priv_user:Privilege {level: "user", description: "Works with user permissions", priority: "low"})
CREATE (priv_optional:Privilege {level: "optional", description: "Root improves results but not required", priority: "medium"});

// ========== SECTION 4: SCAN TYPES (AVEC DÉPENDANCES) ==========
CREATE (scan_syn:Scan {
  name: "SYN Scan",
  code: "-sS",
  description: "TCP SYN (Half-open) - Rapide et stealthy, ne complète pas la connexion",
  requires_root: true,
  speed: "fast",
  stealth: "high",
  detection_capability: "high",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning"
})

CREATE (scan_tcp:Scan {
  name: "TCP Connect Scan",
  code: "-sT",
  description: "Complète la connexion TCP - Plus lent mais détectable et sans privilèges",
  requires_root: false,
  speed: "slow",
  stealth: "low",
  detection_capability: "high",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning"
})

CREATE (scan_udp:Scan {
  name: "UDP Scan",
  code: "-sU",
  description: "Scan des ports UDP - Très lent, nécessite root, utile pour DNS/SNMP/DHCP",
  requires_root: true,
  speed: "very_slow",
  stealth: "medium",
  detection_capability: "medium",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning",
  protocol: "UDP"
})

CREATE (scan_fin:Scan {
  name: "FIN Scan",
  code: "-sF",
  description: "Envoie paquet TCP avec flag FIN - Évite pare-feu basiques",
  requires_root: true,
  speed: "slow",
  stealth: "very_high",
  detection_capability: "medium",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning"
})

CREATE (scan_null:Scan {
  name: "NULL Scan",
  code: "-sN",
  description: "Paquet TCP sans aucun flag - Fonctionne sur Unix uniquement",
  requires_root: true,
  speed: "slow",
  stealth: "very_high",
  detection_capability: "medium",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning"
})

CREATE (scan_xmas:Scan {
  name: "XMAS Scan",
  code: "-sX",
  description: "Flags FIN, PSH, URG activés - Ressemble à un paquet de Noël",
  requires_root: true,
  speed: "slow",
  stealth: "very_high",
  detection_capability: "medium",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "port_scanning"
})

CREATE (scan_ack:Scan {
  name: "ACK Scan",
  code: "-sA",
  description: "Envoie paquets ACK - Détecte pare-feu, ne découvre pas ports ouverts",
  requires_root: true,
  speed: "medium",
  stealth: "high",
  detection_capability: "low",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "firewall_detection"
})

CREATE (scan_window:Scan {
  name: "Window Scan",
  code: "-sW",
  description: "Variante ACK qui examine la fenêtre TCP",
  requires_root: true,
  speed: "medium",
  stealth: "high",
  detection_capability: "low",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "firewall_detection"
})

CREATE (scan_ping:Scan {
  name: "Ping Scan",
  code: "-sn",
  description: "Découverte hôtes sans scan ports - Ping uniquement, très rapide",
  requires_root: false,
  speed: "very_fast",
  stealth: "low",
  detection_capability: "low",
  requires_open_port: false,
  requires_closed_port: false,
  scan_type: "host_discovery"
})

CREATE (scan_os:Scan {
  name: "OS Detection",
  code: "-O",
  description: "Détecte le système d'exploitation",
  requires_root: true,
  speed: "medium",
  stealth: "low",
  detection_capability: "high",
  requires_open_port: true,
  requires_closed_port: true,
  scan_type: "os_detection"
})

CREATE (scan_version:Scan {
  name: "Version Detection",
  code: "-sV",
  description: "Détecte la version des services sur ports ouverts",
  requires_root: false,
  speed: "slow",
  stealth: "low",
  detection_capability: "high",
  requires_open_port: true,
  requires_closed_port: false,
  scan_type: "version_detection"
})

CREATE (scan_aggressive:Scan {
  name: "Aggressive Scan",
  code: "-A",
  description: "Combine -sV -O --script --traceroute - Puissant mais très bruyant",
  requires_root: true,
  speed: "slow",
  stealth: "very_low",
  detection_capability: "very_high",
  requires_open_port: true,
  requires_closed_port: true,
  scan_type: "advanced_scanning"
});

// ========== SECTION 5: OPTIONS ==========
CREATE (opt_ports:Option {
  code: "-p",
  name: "Port Specification",
  description: "Spécifie les ports à scanner",
  category: "port_specification",
  requires_root: false,
  works_without_scan: false
})

CREATE (opt_fast:Option {
  code: "-F",
  name: "Fast Mode",
  description: "Scan les top 100 ports uniquement",
  category: "performance",
  requires_root: false,
  works_without_scan: true
})

CREATE (opt_noping:Option {
  code: "-Pn",
  name: "Skip Ping",
  description: "Assume l'hôte en ligne, ne pas faire de ping préalable",
  category: "discovery",
  requires_root: false,
  works_without_scan: true
})

CREATE (opt_timing_paranoid:Option {
  code: "-T0",
  name: "Paranoid Timing",
  description: "Très lent - Un port à la fois, évite les IDS",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "critical",
  stealth_impact: "very_high"
})

CREATE (opt_timing_sneaky:Option {
  code: "-T1",
  name: "Sneaky Timing",
  description: "Lent et furtif pour éviter les alertes",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "severe",
  stealth_impact: "high"
})

CREATE (opt_timing_polite:Option {
  code: "-T2",
  name: "Polite Timing",
  description: "Modéré, ralentit pour minimiser l'impact",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "moderate",
  stealth_impact: "medium"
})

CREATE (opt_timing_normal:Option {
  code: "-T3",
  name: "Normal Timing",
  description: "Timing par défaut, équilibré",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "none",
  stealth_impact: "low"
})

CREATE (opt_timing_aggressive:Option {
  code: "-T4",
  name: "Aggressive Timing",
  description: "Rapide, assume bonne connexion réseau",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "boost",
  stealth_impact: "very_low"
})

CREATE (opt_timing_insane:Option {
  code: "-T5",
  name: "Insane Timing",
  description: "Très rapide mais peut causer imprécisions",
  category: "timing",
  requires_root: false,
  works_without_scan: true,
  speed_impact: "extreme",
  stealth_impact: "critical"
})

CREATE (opt_os_detection:Option {
  code: "-O",
  name: "OS Detection",
  description: "Active la détection du système d'exploitation",
  category: "detection",
  requires_root: true,
  works_without_scan: false
})

CREATE (opt_version:Option {
  code: "-sV",
  name: "Version Detection",
  description: "Détecte la version des services",
  category: "detection",
  requires_root: false,
  works_without_scan: false
})

CREATE (opt_script:Option {
  code: "--script",
  name: "NSE Scripts",
  description: "Exécute des scripts NSE personnalisés",
  category: "scripting",
  requires_root: false,
  works_without_scan: false
})

CREATE (opt_decoy:Option {
  code: "-D",
  name: "Decoy Scan",
  description: "Utilise adresses leurres pour masquer votre IP",
  category: "evasion",
  requires_root: true,
  works_without_scan: true
})

CREATE (opt_fragmentation:Option {
  code: "-f",
  name: "Fragment Packets",
  description: "Fragmente les paquets pour éviter filtres",
  category: "evasion",
  requires_root: true,
  works_without_scan: true
})

CREATE (opt_mtu:Option {
  code: "--mtu",
  name: "MTU Size",
  description: "Spécifie taille MTU pour fragmentation",
  category: "evasion",
  requires_root: true,
  works_without_scan: true
})

CREATE (opt_source_ip:Option {
  code: "-S",
  name: "Source IP Spoofing",
  description: "Spoofie l'adresse IP source",
  category: "evasion",
  requires_root: true,
  works_without_scan: true
})

CREATE (opt_output_normal:Option {
  code: "-oN",
  name: "Normal Output",
  description: "Format texte normal",
  category: "output",
  requires_root: false,
  works_without_scan: true
})

CREATE (opt_output_xml:Option {
  code: "-oX",
  name: "XML Output",
  description: "Format XML parsable",
  category: "output",
  requires_root: false,
  works_without_scan: true
})

CREATE (opt_output_all:Option {
  code: "-oA",
  name: "All Output Formats",
  description: "Tous formats (normal, XML, grepable)",
  category: "output",
  requires_root: false,
  works_without_scan: true
})

CREATE (opt_verbose:Option {
  code: "-v",
  name: "Verbose Output",
  description: "Mode verbeux, plus de détails (-vv pour très verbeux)",
  category: "output",
  requires_root: false,
  works_without_scan: true
});

// ========== SECTION 6: SCRIPTS NSE ==========
CREATE (script_smb_os:Script {
  code: "smb-os-discovery",
  name: "SMB OS Discovery",
  description: "Détecte OS via SMB",
  risk_level: "low",
  category: "discovery",
  requires_port: 445,
  protocol: "SMB"
})

CREATE (script_smb_shares:Script {
  code: "smb-enum-shares",
  name: "SMB Enum Shares",
  description: "Énumère partages SMB accessibles",
  risk_level: "medium",
  category: "enumeration",
  requires_port: 445,
  protocol: "SMB"
})

CREATE (script_http_title:Script {
  code: "http-title",
  name: "HTTP Title",
  description: "Récupère titre des pages web",
  risk_level: "low",
  category: "discovery",
  requires_port: 80,
  protocol: "HTTP"
})

CREATE (script_ssl_cert:Script {
  code: "ssl-cert",
  name: "SSL Certificate Info",
  description: "Infos du certificat SSL/TLS",
  risk_level: "low",
  category: "discovery",
  requires_port: 443,
  protocol: "HTTPS"
})

CREATE (script_ssh_hostkey:Script {
  code: "ssh-hostkey",
  name: "SSH Hostkey",
  description: "Récupère clé publique SSH",
  risk_level: "low",
  category: "discovery",
  requires_port: 22,
  protocol: "SSH"
});

// ========== SECTION 7: PORT RANGES - RELATION AVEC SERVICES ==========
MATCH (p:PortRange {name: "80"}), (s:Script {requires_port: 80}) CREATE (p)-[:RUNS_SCRIPT]->(s)
MATCH (p:PortRange {name: "443"}), (s:Script {requires_port: 443}) CREATE (p)-[:RUNS_SCRIPT]->(s)
MATCH (p:PortRange {name: "445"}), (s:Script {requires_port: 445}) CREATE (p)-[:RUNS_SCRIPT]->(s)
MATCH (p:PortRange {name: "22"}), (s:Script {requires_port: 22}) CREATE (p)-[:RUNS_SCRIPT]->(s);

// ========== SECTION 8: SCAN -> CATÉGORIES ==========
MATCH (s:Scan {code: "-sS"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sT"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sU"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sF"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sN"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sX"}), (c:Category {name: "Port Scanning"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sA"}), (c:Category {name: "Firewall/IDS Evasion"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sW"}), (c:Category {name: "Firewall/IDS Evasion"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sn"}), (c:Category {name: "Host Discovery"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-O"}), (c:Category {name: "OS Detection"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-sV"}), (c:Category {name: "Version Detection"}) CREATE (s)-[:BELONGS_TO]->(c)
MATCH (s:Scan {code: "-A"}), (c:Category {name: "Advanced Scanning"}) CREATE (s)-[:BELONGS_TO]->(c);

// ========== SECTION 9: PRIVILÈGES -> SCANS ==========
MATCH (p:Privilege {level: "root"}), (s:Scan {requires_root: true}) CREATE (p)-[:REQUIRED_FOR]->(s)
MATCH (p:Privilege {level: "user"}), (s:Scan {requires_root: false}) CREATE (p)-[:REQUIRED_FOR]->(s);

// ========== SECTION 10: DÉPENDANCES - PORTS OUVERTS/FERMÉS ==========
MATCH (s:Scan {requires_open_port: true}) SET s.dependency = "Requires open port"
MATCH (s:Scan {requires_closed_port: true}) SET s.dependency = "Requires closed port"
MATCH (s:Scan {requires_open_port: true, requires_closed_port: true}) SET s.dependency = "Requires open AND closed ports";

// ========== SECTION 11: CONFLITS ENTRE SCANS (CRITIQUE) ==========
MATCH (s1:Scan {code: "-sS"}), (s2:Scan {code: "-sT"}) CREATE (s1)-[:CONFLICTS_WITH {reason: "Cannot use both SYN and Connect scans together"}]->(s2)
MATCH (s1:Scan {code: "-sS"}), (s2:Scan {code: "-sU"}) CREATE (s1)-[:COMPATIBLE_WITH {note: "Different protocols"}]->(s2)
MATCH (s1:Scan {code: "-sF"}), (s2:Scan {code: "-sS"}) CREATE (s1)-[:COMPATIBLE_WITH]->(s2)
MATCH (s1:Scan {code: "-sN"}), (s2:Scan {code: "-sX"}) CREATE (s1)-[:COMPATIBLE_WITH]->(s2)
MATCH (s1:Scan {code: "-sA"}), (s2:Scan {code: "-sS"}) CREATE (s1)-[:COMPATIBLE_WITH {note: "Firewall mapping + Port scanning"}]->(s2);

// ========== SECTION 12: OPTIONS -> SCANS (COMPATIBILITÉ) ==========
MATCH (o:Option {code: "-p"}), (s:Scan {code: "-sS"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-p"}), (s:Scan {code: "-sU"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-p"}), (s:Scan {code: "-sT"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-F"}), (s:Scan {code: "-sS"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-F"}), (s:Scan {code: "-sT"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-O"}), (s:Scan {code: "-sS"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-O"}), (s:Scan {code: "-sT"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-sV"}), (s:Scan {code: "-sS"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-sV"}), (s:Scan {code: "-sT"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "--script"}), (s:Scan {code: "-sV"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "--script"}), (s:Scan {code: "-sS"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-T4"}), (s:Scan {code: "-A"}) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-T0"}), (s:Scan) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-T1"}), (s:Scan) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-Pn"}), (s:Scan) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-D"}), (s:Scan) CREATE (o)-[:WORKS_WITH]->(s)
MATCH (o:Option {code: "-f"}), (s:Scan) CREATE (o)-[:WORKS_WITH]->(s);

// ========== SECTION 13: VALIDATIONS LOGIQUES ==========
// Validation: -sV sans ports ouverts = inefficace
CREATE (validation_sV:Validation {
  name: "Version Detection Validity",
  rule: "-sV requires open ports to be detected first",
  severity: "warning",
  depends_on: "-sV",
  needs: "open_ports"
})

// Validation: -O nécessite ports ouverts ET fermés
CREATE (validation_O:Validation {
  name: "OS Detection Validity",
  rule: "-O requires at least one open and one closed port",
  severity: "warning",
  depends_on: "-O",
  needs: "open_and_closed_ports"
})

// Validation: Timing paranoid est très lent
CREATE (validation_timing:Validation {
  name: "Timing Impact",
  rule: "-T0 and -T1 are extremely slow (hours to days)",
  severity: "info",
  depends_on: "-T0",
  alternative: "-T3 (normal timing)"
});

// ========== SECTION 14: PORT RANGES -> OPTIONS ==========
MATCH (p:PortRange {name: "80,443,22,3389,445"}), (o:Option {code: "-p"}) CREATE (p)-[:SPECIFIED_BY]->(o)
MATCH (p:PortRange {name: "1-65535"}), (o:Option {code: "-p"}) CREATE (p)-[:SPECIFIED_BY]->(o)
MATCH (p:PortRange {name: "1-100"}), (o:Option {code: "-F"}) CREATE (p)-[:USED_BY]->(o);

// ========== SECTION 15: SCANS -> PORT RANGES (RECOMMANDATIONS) ==========
MATCH (s:Scan {code: "-sV"}), (p:PortRange {name: "80,443,22,3389,445"}) CREATE (s)-[:RECOMMENDED_PORTS]->(p)
MATCH (s:Scan {code: "-sS"}), (p:PortRange {name: "1-65535"}) CREATE (s)-[:CAN_SCAN]->(p)
MATCH (s:Scan {code: "-sn"}), (p:PortRange {name: "80"}) CREATE (s)-[:IGNORES]->(p);

// ========== SECTION 16: SCRIPTS -> SCANS (DÉPENDANCES) ==========
MATCH (scr:Script {code: "smb-os-discovery"}), (s:Scan {code: "-sV"}) CREATE (scr)-[:REQUIRES_SCAN]->(s)
MATCH (scr:Script {code: "smb-enum-shares"}), (s:Scan {code: "-sV"}) CREATE (scr)-[:REQUIRES_SCAN]->(s)
MATCH (scr:Script {code: "http-title"}), (s:Scan {code: "-sS"}) CREATE (scr)-[:REQUIRES_SCAN]->(s)
MATCH (scr:Script {code: "ssl-cert"}), (s:Scan {code: "-sV"}) CREATE (scr)-[:REQUIRES_SCAN]->(s)
MATCH (scr:Script {code: "ssh-hostkey"}), (s:Scan {code: "-sV"}) CREATE (scr)-[:REQUIRES_SCAN]->(s);

// ========== SECTION 17: EXEMPLES DE COMMANDES ==========
CREATE (ex1:Example {
  name: "Basic SYN Scan",
  command: "nmap -sS 192.168.1.0/24",
  description: "Scan SYN simple sur un réseau entier",
  difficulty: "easy",
  use_case: "Reconnaissance générale",
  estimated_time: "5-10 minutes"
})

CREATE (ex2:Example {
  name: "Full Port Scan with Version Detection",
  command: "nmap -p- -sV -O 192.168.1.1",
  description: "Scan tous les ports, détecte OS et versions",
  difficulty: "medium",
  use_case: "Analyse approfondie d'un hôte",
  estimated_time: "30-60 minutes",
  requires_open_port: true,
  requires_closed_port: true
})

CREATE (ex3:Example {
  name: "UDP Service Discovery",
  command: "nmap -sU -p 53,123,161 192.168.1.1",
  description: "Scan UDP sur ports DNS, NTP, SNMP",
  difficulty: "medium",
  use_case: "Découverte services UDP",
  estimated_time: "10-20 minutes"
})

CREATE (ex4:Example {
  name: "Aggressive Full Scan",
  command: "nmap -A -T4 -p- 192.168.1.1",
  description: "Scan agressif avec tous les détails, timing rapide",
  difficulty: "hard",
  use_case: "Test de pénétration complet",
  estimated_time: "1-2 hours",
  stealth_level: "very_low"
})

CREATE (ex5:Example {
  name: "Stealth Scan",
  command: "nmap -sF -D 192.168.1.2,192.168.1.3 -T0 192.168.1.1",
  description: "Scan FIN avec leurres et timing paranoid",
  difficulty: "hard",
  use_case: "Contournement pare-feu/IDS",
  estimated_time: "several_hours",
  stealth_level: "very_high"
})

CREATE (ex6:Example {
  name: "Host Discovery Only",
  command: "nmap -sn 192.168.1.0/24",
  description: "Découverte rapide hôtes du réseau",
  difficulty: "easy",
  use_case: "Reconnaissance réseau",
  estimated_time: "1-2 minutes"
})

CREATE (ex7:Example {
  name: "SMB Enumeration",
  command: "nmap -sV --script=smb-os-discovery,smb-enum-shares 192.168.1.1",
  description: "Énumération complète SMB",
  difficulty: "medium",
  use_case: "Découverte partages/infos SMB",
  estimated_time: "5-10 minutes"
})

CREATE (ex8:Example {
  name: "Save All Formats",
  command: "nmap -A -oA scan_results 192.168.1.1",
  description: "Scan complet sauvegardé en tous formats",
  difficulty: "easy",
  use_case: "Rapport d'analyse documentée",
  estimated_time: "varies"
});

// ========== SECTION 18: EXEMPLES -> SCANS ==========
MATCH (e:Example {name: "Basic SYN Scan"}), (s:Scan {code: "-sS"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "Full Port Scan with Version Detection"}), (s:Scan {code: "-sV"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "Full Port Scan with Version Detection"}), (s:Scan {code: "-O"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "UDP Service Discovery"}), (s:Scan {code: "-sU"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "Aggressive Full Scan"}), (s:Scan {code: "-A"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "Stealth Scan"}), (s:Scan {code: "-sF"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "Host Discovery Only"}), (s:Scan {code: "-sn"}) CREATE (e)-[:USES_SCAN]->(s)
MATCH (e:Example {name: "SMB Enumeration"}), (s:Scan {code: "-sV"}) CREATE (e)-[:USES_SCAN]->(s);

// ========== SECTION 19: EXEMPLES -> OPTIONS ==========
MATCH (e:Example {name: "Basic SYN Scan"}), (o:Option {code: "-sS"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Full Port Scan with Version Detection"}), (o:Option {code: "-p"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Full Port Scan with Version Detection"}), (o:Option {code: "-sV"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Full Port Scan with Version Detection"}), (o:Option {code: "-O"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "UDP Service Discovery"}), (o:Option {code: "-p"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Aggressive Full Scan"}), (o:Option {code: "-T4"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Stealth Scan"}), (o:Option {code: "-T0"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Stealth Scan"}), (o:Option {code: "-D"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "SMB Enumeration"}), (o:Option {code: "--script"}) CREATE (e)-[:USES_OPTION]->(o)
MATCH (e:Example {name: "Save All Formats"}), (o:Option {code: "-oA"}) CREATE (e)-[:USES_OPTION]->(o);

// ========== SECTION 20: VALIDATIONS LOGIQUES - RELATIONS ==========
MATCH (v:Validation {depends_on: "-sV"}), (s:Scan {code: "-sV"}) CREATE (v)-[:VALIDATES]->(s)
MATCH (v:Validation {depends_on: "-O"}), (s:Scan {code: "-O"}) CREATE (v)-[:VALIDATES]->(s)
MATCH (v:Validation {depends_on: "-T0"}), (o:Option {code: "-T0"}) CREATE (v)-[:VALIDATES]->(o);

// ========== SECTION 21: INCOMPATIBILITÉS TIMING ==========
// Timing paranoid (-T0) très incompatible avec scan rapide
MATCH (o1:Option {code: "-T0"}), (o2:Option {code: "-T5"}) 
CREATE (o1)-[:CONFLICTS_WITH {reason: "Paranoid timing conflicts with Insane timing"}]->(o2)

// Timing agressif avec Stealthy = contradiction
MATCH (o1:Option {code: "-T4"}), (s:Scan {stealth: "very_high"}) 
CREATE (o1)-[:CONFLICTS_WITH {reason: "Aggressive timing defeats stealth scans"}]->(s);

// ========== SECTION 22: INEFFICACITÉS - WARNING VALIDATIONS ==========
// Warning: Version detection sans ports ouverts
CREATE (warning_version:ValidationWarning {
  name: "Version Detection Ineffective",
  message: "-sV without open ports = no results",
  severity: "warning",
  affected_option: "-sV",
  suggested_action: "Ensure ports are open before using -sV"
})

// Warning: ACK scan ne détecte pas ports ouverts
CREATE (warning_ack:ValidationWarning {
  name: "ACK Scan Limitations",
  message: "-sA does NOT discover open ports, only firewall rules",
  severity: "info",
  affected_option: "-sA",
  suggested_action: "Use -sS or -sT for port discovery"
})

// Warning: Aggressive scan très détectable
CREATE (warning_aggressive:ValidationWarning {
  name: "Aggressive Detection",
  message: "-A is very loud and will trigger alerts",
  severity: "critical",
  affected_option: "-A",
  suggested_action: "Use stealthy options for covert scanning"
});

// ========== SECTION 23: METADATA - COMPLEXITÉ ==========
CREATE (complexity_easy:Complexity {level: "EASY", description: "Simple reconnaissance", time_to_learn: "< 1 day", examples: 2})
CREATE (complexity_medium:Complexity {level: "MEDIUM", description: "Intermediate scanning", time_to_learn: "2-3 days", examples: 3})
CREATE (complexity_hard:Complexity {level: "HARD", description: "Advanced techniques", time_to_learn: "> 1 week", examples: 3});

// ========== SECTION 24: SCAN -> COMPLEXITY ==========
MATCH (s:Scan {code: "-sn"}), (c:Complexity {level: "EASY"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-sS"}), (c:Complexity {level: "EASY"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-sV"}), (c:Complexity {level: "MEDIUM"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-sU"}), (c:Complexity {level: "MEDIUM"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-O"}), (c:Complexity {level: "HARD"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-A"}), (c:Complexity {level: "HARD"}) CREATE (s)-[:HAS_COMPLEXITY]->(c)
MATCH (s:Scan {code: "-sF"}), (c:Complexity {level: "HARD"}) CREATE (s)-[:HAS_COMPLEXITY]->(c);

// ========== SECTION 25: VÉRIFICATIONS FINALES ==========
// Count all nodes
MATCH (n) RETURN labels(n) as type, COUNT(*) as count ORDER BY count DESC;

// Count all relationships
MATCH ()-[r]->() RETURN type(r) as relationship, COUNT(*) as count ORDER BY count DESC;

// Show overall graph statistics
MATCH (n) WITH COUNT(DISTINCT n) as nodes
MATCH ()-[r]->() WITH COUNT(DISTINCT r) as rels, nodes
RETURN "🎯 NMAP-AI Knowledge Graph" as status, nodes, rels, "Ready for traversal" as state;

// ========== SECTION 26: EXEMPLE REQUÊTE TRAVERSÉE ==========
// Trouver tous les scans et leurs dépendances
MATCH (s:Scan)
OPTIONAL MATCH (p:Privilege {level: "root"})-[:REQUIRED_FOR]->(s)
OPTIONAL MATCH (s)-[:CONFLICTS_WITH]->(conflict:Scan)
OPTIONAL MATCH (s)-[:HAS_COMPLEXITY]->(c:Complexity)
RETURN s.code, s.name, p.level as privilege, c.level as complexity, COUNT(DISTINCT conflict) as conflicts
ORDER BY s.code;