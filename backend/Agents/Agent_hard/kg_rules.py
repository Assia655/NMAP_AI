"""
Lightweight, local KG-style rule checks for Nmap options.

This is a placeholder for future Neo4j-backed KG lookups. The functions here
are intentionally simple and safe to run offline; they return both a boolean
and a human-readable message to inform the diffusion scoring/validation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import re


def check_requires_root(scan_types: List[str], os_detection: bool) -> Tuple[bool, str]:
    """Return True if the combination likely needs root."""
    if ("-sS" in scan_types) or ("-sU" in scan_types) or os_detection:
        return True, "Raw socket/OS detection likely needs sudo/root."
    return False, ""


def check_port_format(ports: str | None) -> Tuple[bool, str]:
    """Validate port string format; accept ranges and CSV."""
    if ports is None:
        return True, ""
    p = ports.strip()
    if re.fullmatch(r"(\d{1,5}(-\d{1,5})?)(,\d{1,5}(-\d{1,5})?)*", p):
        return True, ""
    return False, f"Ports format looks invalid: {ports}"


def check_conflicts(scan_types: List[str], timing: str | None) -> List[str]:
    """Simple incompatibility/unsafe hints."""
    issues: List[str] = []
    if "-sU" in scan_types and "-sS" not in scan_types:
        issues.append("UDP-only scans are slow; consider pairing with -sS or limiting ports.")
    if timing and timing not in ("-T0", "-T1", "-T2", "-T3", "-T4", "-T5"):
        issues.append(f"Timing template {timing} is not valid.")
    return issues


def summarize_command(parts: List[str]) -> str:
    """Turn a command parts list into a short description string."""
    return " ".join(parts)


def intent_coverage_score(query: str, command_summary: str) -> float:
    """
    Very cheap similarity: token overlap ratio between query and command summary.
    Returns 0..1.
    """
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1}
    c_tokens = {t for t in re.findall(r"[a-z0-9]+", command_summary.lower()) if len(t) > 1}
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / len(q_tokens | c_tokens)
