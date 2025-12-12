from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import re

from .kg_rules import (
    check_conflicts,
    check_port_format,
    check_requires_root,
    intent_coverage_score,
    summarize_command,
)


# -----------------------------
# Utils
# -----------------------------
def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _has_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Command state (symbolic diffusion state)
# -----------------------------
@dataclass
class NmapState:
    scan_types: List[str] = field(default_factory=list)  # e.g., ["-sS", "-sU"]
    ports: Optional[str] = None                          # e.g., "1-1000" or "80,443"
    timing: Optional[str] = None                         # -T0..-T5
    os_detection: bool = False                           # -O
    service_version: bool = False                        # -sV
    scripts: List[str] = field(default_factory=list)     # --script=...
    ping_disable: bool = False                           # -Pn
    dns_disable: bool = False                            # -n
    retries: Optional[int] = None                        # --max-retries
    host_timeout: Optional[str] = None                   # --host-timeout
    min_rate: Optional[int] = None                       # --min-rate
    max_rate: Optional[int] = None                       # --max-rate
    fragment: bool = False                               # -f
    decoys: Optional[str] = None                         # -D RND:10 or list
    spoof_mac: Optional[str] = None                      # --spoof-mac
    iface: Optional[str] = None                          # -e eth0
    source_port: Optional[int] = None                    # --source-port
    verbosity: int = 0                                   # -v / -vv
    extra: List[str] = field(default_factory=list)       # any safe extras
    requires_root: bool = False                          # sudo needed
    notes: List[str] = field(default_factory=list)       # step notes (diffusion trace)

    def clone(self) -> "NmapState":
        return NmapState(
            scan_types=list(self.scan_types),
            ports=self.ports,
            timing=self.timing,
            os_detection=self.os_detection,
            service_version=self.service_version,
            scripts=list(self.scripts),
            ping_disable=self.ping_disable,
            dns_disable=self.dns_disable,
            retries=self.retries,
            host_timeout=self.host_timeout,
            min_rate=self.min_rate,
            max_rate=self.max_rate,
            fragment=self.fragment,
            decoys=self.decoys,
            spoof_mac=self.spoof_mac,
            iface=self.iface,
            source_port=self.source_port,
            verbosity=self.verbosity,
            extra=list(self.extra),
            requires_root=self.requires_root,
            notes=list(self.notes),
        )


# -----------------------------
# Diffusion-like Hard Agent
# -----------------------------
class HardDiffusionAgent:
    """
    HardDiffusionAgent
    - "Diffusion-like" approach: start from a noisy / imperfect Nmap state,
      then iteratively refine it by applying constraints, resolving conflicts,
      and optimizing toward the user's intent.
    - Output: dict with final command, reasoning steps, warnings.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # --------- Public API ---------
    def generate(self, query: str, target: str) -> Dict[str, Any]:
        query = _norm_spaces(query)
        target = _norm_spaces(target)

        if not query:
            return {"ok": False, "error": "Empty query."}
        if not target:
            return {"ok": False, "error": "Empty target."}

        intent = self._parse_intent(query)

        # Try a few stochastic attempts; keep best score
        attempts = 3
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for _ in range(attempts):
            candidate = self._run_diffusion(query, target, intent)
            score_val = candidate["score"]["total"]
            if score_val > best_score:
                best_score = score_val
                best = candidate
            if score_val >= 0.85:  # good enough, stop early
                break

        assert best is not None
        return best

    # --------- Single diffusion attempt ---------
    def _run_diffusion(self, query: str, target: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        # 1) forward/noise init (noisy candidate)
        state = self._noisy_init(intent)

        # 2) reverse/denoise: iterative refinement
        steps = intent["steps"]
        warnings: List[str] = []
        for t in range(1, steps + 1):
            state, w = self._denoise_step(state, intent, t, steps)
            warnings.extend(w)

        # 3) final validation & build command
        valid, final_warnings = self._validate_state(state, intent)
        warnings.extend(final_warnings)

        cmd, summary_parts = self._build_command(state, target, with_parts=True)
        score = self._score(query, summary_parts, state)
        repaired = score["total"] < 0.85

        return {
            "ok": True,
            "agent": "hard_diffusion",
            "target": target,
            "query": query,
            "intent": intent,
            "valid": valid,
            "command": cmd,
            "requires_root": state.requires_root,
            "warnings": self._dedupe(warnings),
            "diffusion_trace": state.notes,
            "score": score,
            "repaired": repaired,
        }

    # --------- Intent parsing ---------
    def _parse_intent(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        wants_udp = _has_any(q, ["udp", " -su", "s_u", "s-u"])
        wants_tcp = _has_any(q, ["tcp", " -ss", "syn", "stealth", "s s", "s-s", "-sS".lower()]) or True

        wants_os = _has_any(q, ["os detection", "detect os", "os", "-o", " -o "]) and not _has_any(q, ["no os", "without os"])
        wants_sv = _has_any(q, ["service", "version", "banner", "-sv", "-sV".lower()])
        wants_full = _has_any(q, ["full", "all ports", "1-65535", "65535"])

        stealth = _has_any(q, ["stealth", "quiet", "low noise", "discreet", "evasion", "ids", "avoid detection"])
        aggressive = _has_any(q, ["aggressive", "fast", "speed", "t5", "-t5", "rapid"])

        # Decide timing (conflicts handled later)
        timing = None
        if stealth and not aggressive:
            timing = self._rng.choice(["-T1", "-T2"])
        elif aggressive and not stealth:
            timing = self._rng.choice(["-T4", "-T5"])
        elif stealth and aggressive:
            timing = self._rng.choice(["-T2", "-T3"])  # compromise

        ports = None
        if wants_full:
            ports = "1-65535"
        else:
            # If query hints top ports
            if _has_any(q, ["top", "common", "default"]):
                ports = None  # leave to nmap default
            else:
                # mild default for "hard" intents: scan more than top 1000 often
                ports = self._rng.choice(["1-2000", "1-5000", None])

        steps = 7 if (stealth or wants_udp or wants_os or wants_sv) else 5

        return {
            "wants_tcp": wants_tcp,
            "wants_udp": wants_udp,
            "wants_os": wants_os,
            "wants_sv": wants_sv,
            "ports": ports,
            "stealth": stealth,
            "aggressive": aggressive,
            "timing": timing,
            "steps": steps,
        }

    # --------- Forward / Noise init ---------
    def _noisy_init(self, intent: Dict[str, Any]) -> NmapState:
        s = NmapState()

        # Start intentionally imperfect / noisy
        # Sometimes missing TCP, or has conflicting timing, etc.
        if self._rng.random() < 0.6:
            s.scan_types.append("-sU" if intent["wants_udp"] else "-sS")
        else:
            s.scan_types.append("-sS")

        if intent["wants_udp"] and self._rng.random() < 0.5:
            # sometimes start only with UDP -> will be corrected
            s.scan_types = ["-sU"]

        # add "noisy" timing (maybe too fast for stealth)
        if intent["timing"]:
            s.timing = intent["timing"]
        else:
            s.timing = self._rng.choice([None, "-T4", "-T5", "-T2"])

        s.os_detection = intent["wants_os"] and (self._rng.random() < 0.7)
        s.service_version = intent["wants_sv"] and (self._rng.random() < 0.7)

        s.ports = intent["ports"] if self._rng.random() < 0.8 else None

        # noisy defaults
        s.ping_disable = self._rng.random() < 0.4
        s.dns_disable = self._rng.random() < 0.5
        s.retries = self._rng.choice([None, 2, 3, 5])
        s.verbosity = self._rng.choice([0, 1])

        s.notes.append("t=0 init noisy state")
        return s

    # --------- Reverse / Denoise step ---------
    def _denoise_step(self, state: NmapState, intent: Dict[str, Any], t: int, T: int) -> Tuple[NmapState, List[str]]:
        s = state.clone()
        warnings: List[str] = []

        # --- (A) Ensure required scan types ---
        if intent["wants_tcp"] and "-sS" not in s.scan_types:
            s.scan_types.append("-sS")
            s.notes.append(f"t={t}: add TCP SYN scan (-sS)")

        if intent["wants_udp"] and "-sU" not in s.scan_types:
            # Add UDP gradually (later steps) to reduce noise and avoid early conflicts
            if t >= max(2, T // 3):
                s.scan_types.append("-sU")
                s.notes.append(f"t={t}: add UDP scan (-sU)")

        # --- (B) Conflict resolution: stealth vs timing ---
        if intent["stealth"]:
            if s.timing in ("-T4", "-T5"):
                s.timing = "-T2"
                s.notes.append(f"t={t}: reduce timing for stealth (-T2)")
                warnings.append("Stealth requested: downgraded timing from -T4/-T5 to -T2.")
            if s.min_rate is not None and s.min_rate > 100:
                s.min_rate = 50
                s.notes.append(f"t={t}: lower --min-rate for stealth")
        elif intent["aggressive"]:
            if s.timing in (None, "-T0", "-T1", "-T2"):
                s.timing = "-T4"
                s.notes.append(f"t={t}: increase timing for speed (-T4)")

        # --- (C) OS detection / service version: add carefully ---
        if intent["wants_os"] and not s.os_detection and t >= max(2, T // 2):
            s.os_detection = True
            s.notes.append(f"t={t}: enable OS detection (-O)")

        if intent["wants_sv"] and not s.service_version and t >= max(2, T // 2):
            s.service_version = True
            s.notes.append(f"t={t}: enable service/version detection (-sV)")

        # --- (D) Add safety + stability knobs ---
        # Reduce DNS noise
        if intent["stealth"] and not s.dns_disable:
            s.dns_disable = True
            s.notes.append(f"t={t}: disable DNS resolution (-n)")

        # Avoid host discovery if behind firewall (common in stealth)
        if intent["stealth"] and not s.ping_disable and t >= max(2, T // 2):
            s.ping_disable = True
            s.notes.append(f"t={t}: disable ping (-Pn)")

        # Retries for stability (stealth often prefers fewer retries to limit noise)
        if intent["stealth"]:
            if s.retries is None or s.retries > 3:
                s.retries = 2
                s.notes.append(f"t={t}: set --max-retries 2 (stealth)")
        else:
            if s.retries is None:
                s.retries = 3
                s.notes.append(f"t={t}: set --max-retries 3")

        # Add host timeout near the end for hard cases
        if t >= T - 1 and s.host_timeout is None:
            s.host_timeout = "60s" if intent["stealth"] else "30s"
            s.notes.append(f"t={t}: set --host-timeout {s.host_timeout}")

        # --- (E) Root requirement inference ---
        if ("-sS" in s.scan_types) or ("-sU" in s.scan_types) or s.os_detection:
            s.requires_root = True

        # --- (F) Gentle verbosity for debugging (optional) ---
        if t == T and s.verbosity < 1:
            s.verbosity = 1
            s.notes.append(f"t={t}: set verbosity -v")

        return s, warnings

    # --------- Validation (constraints) ---------
    def _validate_state(self, s: NmapState, intent: Dict[str, Any]) -> Tuple[bool, List[str]]:
        warnings: List[str] = []

        # Remove duplicates in scan types
        s.scan_types = self._dedupe(s.scan_types)

        # Conflicting scan types sanity
        if "-sS" not in s.scan_types and intent["wants_tcp"]:
            warnings.append("TCP scan requested but -sS missing; adding it.")
            s.scan_types.append("-sS")

        # Timing sanity
        if s.timing and s.timing not in ("-T0", "-T1", "-T2", "-T3", "-T4", "-T5"):
            warnings.append(f"Invalid timing {s.timing} detected; removing it.")
            s.timing = None

        # Ports sanity (basic + kg_rules)
        ok_ports, msg = check_port_format(s.ports)
        if not ok_ports:
            warnings.append(msg)
            s.ports = None

        # Root requirement note
        if s.requires_root:
            warnings.append("This scan likely requires elevated privileges (sudo).")

        # If stealth requested, avoid ultra-aggressive options
        if intent["stealth"] and s.timing in ("-T4", "-T5"):
            warnings.append("Stealth requested but timing too high; adjusted to -T2.")
            s.timing = "-T2"

        # KG-style conflict hints
        warnings.extend(check_conflicts(s.scan_types, s.timing))

        valid = True
        return valid, warnings

    # --------- Build final command string ---------
    def _build_command(self, s: NmapState, target: str, with_parts: bool = False) -> Any:
        parts: List[str] = []

        if s.requires_root:
            parts.append("sudo")

        parts.append("nmap")

        # scan types
        parts.extend(s.scan_types)

        # ports
        if s.ports:
            parts.extend(["-p", s.ports])

        # feature flags
        if s.os_detection:
            parts.append("-O")
        if s.service_version:
            parts.append("-sV")

        # ping / dns
        if s.ping_disable:
            parts.append("-Pn")
        if s.dns_disable:
            parts.append("-n")

        # timing
        if s.timing:
            parts.append(s.timing)

        # retries + timeouts
        if s.retries is not None:
            parts.extend(["--max-retries", str(s.retries)])
        if s.host_timeout:
            parts.extend(["--host-timeout", s.host_timeout])

        # rates
        if s.min_rate is not None:
            parts.extend(["--min-rate", str(s.min_rate)])
        if s.max_rate is not None:
            parts.extend(["--max-rate", str(s.max_rate)])

        # evasions (kept optional, not forced)
        if s.fragment:
            parts.append("-f")
        if s.decoys:
            parts.extend(["-D", s.decoys])
        if s.spoof_mac:
            parts.extend(["--spoof-mac", s.spoof_mac])
        if s.source_port is not None:
            parts.extend(["--source-port", str(s.source_port)])
        if s.iface:
            parts.extend(["-e", s.iface])

        # scripts
        for sc in s.scripts:
            if sc.strip():
                parts.append(f"--script={sc.strip()}")

        # verbosity
        if s.verbosity == 1:
            parts.append("-v")
        elif s.verbosity >= 2:
            parts.append("-vv")

        # extras (safe)
        parts.extend(s.extra)

        parts.append(target)

        cmd = _norm_spaces(" ".join(parts))
        if with_parts:
            return cmd, parts
        return cmd

    # --------- Scoring ---------
    def _score(self, query: str, cmd_parts: List[str], state: NmapState) -> Dict[str, float]:
        """Combine intent coverage and constraint hints into a 0..1 score."""
        summary = summarize_command(cmd_parts)
        intent_cov = intent_coverage_score(query, summary)

        requires_root, _ = check_requires_root(state.scan_types, state.os_detection)
        constraint_hits = 1.0
        if state.timing and state.timing not in ("-T0", "-T1", "-T2", "-T3", "-T4", "-T5"):
            constraint_hits -= 0.2

        total = _clamp(0.5 * intent_cov + 0.5 * constraint_hits, 0.0, 1.0)

        return {
            "total": total,
            "intent": intent_cov,
            "constraints": constraint_hits,
            "requires_root": 1.0 if requires_root else 0.0,
        }

    # --------- Helpers ---------
    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out
