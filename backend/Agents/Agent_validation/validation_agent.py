#!/usr/bin/env python3
"""
NMAP Command Validation Agent
Pure execution-based validation without knowledge graphs or rule databases.
Treats Nmap itself as the ground-truth oracle.
"""

import subprocess
import re
import json
import time
import os
import shlex
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ValidationStatus(Enum):
    """Validation outcome status"""
    VALID = "valid"
    INVALID = "invalid"
    REPAIRABLE = "repairable"
    UNSAFE = "unsafe"
    PRIVILEGE_REQUIRED = "privilege_required"


@dataclass
class ValidationResult:
    """Structured validation result"""
    status: str
    command: str
    is_valid: bool
    is_safe: bool
    requires_privilege: bool
    execution_time: float
    exit_code: int
    stdout: str
    stderr: str
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metadata: Dict


class NmapValidator:
    """
    Execution-based Nmap command validator.
    No knowledge graphs, no rule databases - pure oracle-based validation.
    """

    # Safe targets for validation
    SAFE_TARGETS = ['127.0.0.1', 'localhost', '::1']

    # Dangerous patterns to block
    UNSAFE_PATTERNS = [
        r'--script.*exploit',
        r'--script.*dos',
        r'--script.*brute',
        r'-sC.*--script.*auth',
        r'--max-rate\s+[0-9]{4,}',  # Rate > 1000
    ]

    # Dangerous targets to prevent
    FORBIDDEN_TARGETS = [
        r'0\.0\.0\.0/0',
        r'::/0',
        r'224\.',  # Multicast
        r'255\.255\.255\.255',  # Broadcast
    ]

    def __init__(self, timeout: int = 30, nmap_path: str = 'nmap'):
        """
        Initialize validator.

        Args:
            timeout: Maximum execution time in seconds
            nmap_path: Path to nmap binary
        """
        self.timeout = timeout
        self.nmap_path = nmap_path
        self._verify_nmap_available()

    def _verify_nmap_available(self):
        """Verify nmap is installed and accessible"""
        try:
            result = subprocess.run(
                [self.nmap_path, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Nmap not available or not functioning")
        except FileNotFoundError:
            raise RuntimeError(f"Nmap not found at: {self.nmap_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Nmap version check timed out")

    def validate(self, command: str) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            command: Nmap command string to validate

        Returns:
            ValidationResult object with complete validation details
        """
        errors = []
        warnings = []
        suggestions = []
        metadata = {}

        # Step 1: Parse command
        parsed = self._parse_command(command)
        if not parsed['valid']:
            return self._build_result(
                status=ValidationStatus.INVALID,
                command=command,
                errors=parsed['errors'],
                metadata=parsed
            )

        # Step 2: Safety gate
        safety_check = self._check_safety(parsed)
        if not safety_check['safe']:
            return self._build_result(
                status=ValidationStatus.UNSAFE,
                command=command,
                errors=safety_check['errors'],
                warnings=safety_check['warnings'],
                metadata=safety_check
            )

        # Step 3: Replace target with safe localhost
        safe_command = self._make_safe_command(parsed)

        # Step 4: Execute and capture
        exec_result = self._execute_command(safe_command)

        # Step 5: Analyze results
        analysis = self._analyze_execution(exec_result, parsed)

        # Step 6: Determine final status
        final_status = self._determine_status(analysis)

        return self._build_result(
            status=final_status,
            command=command,
            execution_time=exec_result['duration'],
            exit_code=exec_result['exit_code'],
            stdout=exec_result['stdout'],
            stderr=exec_result['stderr'],
            errors=analysis['errors'],
            warnings=analysis['warnings'],
            suggestions=analysis['suggestions'],
            metadata={
                'parsed': parsed,
                'execution': exec_result,
                'analysis': analysis
            }
        )

    def _parse_command(self, command: str) -> Dict:
        """
        Parse nmap command into components.

        Args:
            command: Raw command string

        Returns:
            Dictionary with parsed components and validation status
        """
        result = {
            'valid': False,
            'errors': [],
            'raw': command,
            'binary': None,
            'flags': [],
            'options': {},
            'targets': []
        }

        try:
            # Remove leading/trailing whitespace
            command = command.strip()

            # Split command safely
            parts = shlex.split(command)

            if not parts:
                result['errors'].append("Empty command")
                return result

            # First part should be nmap or include nmap
            if 'nmap' not in parts[0].lower():
                result['errors'].append(f"Not an nmap command: {parts[0]}")
                return result

            result['binary'] = parts[0]

            # Parse remaining parts
            i = 1
            while i < len(parts):
                part = parts[i]

                # Check if it's a flag (starts with -)
                if part.startswith('-'):
                    # Check if next part is a value for this flag
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        # This is an option with value
                        result['options'][part] = parts[i + 1]
                        i += 2
                    else:
                        # This is a standalone flag
                        result['flags'].append(part)
                        i += 1
                else:
                    # This is a target
                    result['targets'].append(part)
                    i += 1

            if not result['targets']:
                result['errors'].append("No target specified")
                return result

            result['valid'] = True

        except Exception as e:
            result['errors'].append(f"Parse error: {str(e)}")

        return result

    def _check_safety(self, parsed: Dict) -> Dict:
        """
        Pre-flight safety checks.

        Args:
            parsed: Parsed command dictionary

        Returns:
            Safety check results
        """
        result = {
            'safe': True,
            'errors': [],
            'warnings': []
        }

        full_command = ' '.join([parsed['binary']] + parsed['flags'] +
                                [f"{k} {v}" for k, v in parsed['options'].items()] +
                                parsed['targets'])

        # Check for unsafe patterns
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, full_command, re.IGNORECASE):
                result['safe'] = False
                result['errors'].append(f"Unsafe pattern detected: {pattern}")

        # Check for forbidden targets
        for target in parsed['targets']:
            for forbidden in self.FORBIDDEN_TARGETS:
                if re.match(forbidden, target):
                    result['safe'] = False
                    result['errors'].append(f"Forbidden target: {target}")

        # Check for timing attacks
        if any('--max-rate' in str(k) for k in parsed['options'].keys()):
            result['warnings'].append("High scan rate detected - may be aggressive")

        # Check for script engine usage
        if any('--script' in flag or '--script' in opt
               for flag in parsed['flags']
               for opt in parsed['options'].keys()):
            result['warnings'].append("NSE scripts detected - review for safety")

        return result

    def _make_safe_command(self, parsed: Dict) -> str:
        """
        Replace targets with safe localhost for validation.

        Args:
            parsed: Parsed command dictionary

        Returns:
            Safe command string
        """
        # Replace all targets with 127.0.0.1
        safe_parts = [self.nmap_path]
        safe_parts.extend(parsed['flags'])

        for opt, val in parsed['options'].items():
            safe_parts.append(opt)
            safe_parts.append(val)

        # Use localhost as safe target
        safe_parts.append('127.0.0.1')

        return ' '.join(safe_parts)

    def _execute_command(self, command: str) -> Dict:
        """
        Execute nmap command and capture results.

        Args:
            command: Command to execute

        Returns:
            Execution results including timing and outputs
        """
        result = {
            'command': command,
            'exit_code': -1,
            'stdout': '',
            'stderr': '',
            'duration': 0.0,
            'timed_out': False,
            'error': None
        }

        start_time = time.time()

        try:
            # Execute command with timeout
            proc = subprocess.run(
                shlex.split(command),
                capture_output=True,
                timeout=self.timeout,
                text=True
            )

            result['exit_code'] = proc.returncode
            result['stdout'] = proc.stdout
            result['stderr'] = proc.stderr

        except subprocess.TimeoutExpired:
            result['timed_out'] = True
            result['error'] = f"Command timed out after {self.timeout}s"

        except Exception as e:
            result['error'] = str(e)

        finally:
            result['duration'] = time.time() - start_time

        return result

    def _analyze_execution(self, exec_result: Dict, parsed: Dict) -> Dict:
        """
        Analyze execution results to determine validity.

        Args:
            exec_result: Results from execution
            parsed: Parsed command structure

        Returns:
            Analysis with errors, warnings, and suggestions
        """
        analysis = {
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'requires_privilege': False,
            'syntax_valid': True,
            'executable': True
        }

        stderr = exec_result['stderr'].lower()
        stdout = exec_result['stdout'].lower()
        exit_code = exec_result['exit_code']

        # Check for timeout
        if exec_result['timed_out']:
            analysis['errors'].append("Command timed out - may be too slow or hung")
            analysis['executable'] = False

        # Check exit code
        if exit_code != 0:
            analysis['executable'] = False
            analysis['errors'].append(f"Non-zero exit code: {exit_code}")

        # Parse stderr for specific error patterns

        # Privilege errors
        privilege_patterns = [
            'requires root',
            'operation not permitted',
            'must be root',
            'you need to be root',
            'insufficient privileges'
        ]

        for pattern in privilege_patterns:
            if pattern in stderr:
                analysis['requires_privilege'] = True
                analysis['errors'].append("Requires elevated privileges (root/sudo)")
                analysis['suggestions'].append("Run with sudo or as root user")

        # Invalid option errors
        invalid_option_patterns = [
            r'invalid option',
            r'unrecognized option',
            r'unknown option',
            r'illegal option'
        ]

        for pattern in invalid_option_patterns:
            if re.search(pattern, stderr):
                analysis['syntax_valid'] = False
                analysis['errors'].append("Invalid or unknown option detected")

                # Try to extract the specific option
                match = re.search(r'[\'"](-+[a-zA-Z0-9-]+)[\'"]', exec_result['stderr'])
                if match:
                    analysis['suggestions'].append(f"Check option: {match.group(1)}")

        # Conflicting options
        conflict_patterns = [
            'conflicting',
            'cannot be used with',
            'incompatible',
            'mutually exclusive'
        ]

        for pattern in conflict_patterns:
            if pattern in stderr:
                analysis['errors'].append("Conflicting options detected")
                analysis['suggestions'].append("Review option compatibility")

        # Target errors
        if 'failed to resolve' in stderr or 'unknown host' in stderr:
            analysis['warnings'].append("Target resolution failed (expected for 127.0.0.1 test)")

        # Parse warnings
        if 'warning' in stderr or 'warning' in stdout:
            analysis['warnings'].append("Nmap issued warnings - review output")

        # Successful execution indicators
        if exit_code == 0 and 'starting nmap' in stdout:
            analysis['suggestions'].append("Command syntax is valid and executable")

        return analysis

    def _determine_status(self, analysis: Dict) -> ValidationStatus:
        """
        Determine final validation status.

        Args:
            analysis: Analysis results

        Returns:
            ValidationStatus enum value
        """
        if not analysis['syntax_valid']:
            return ValidationStatus.INVALID

        if analysis['requires_privilege']:
            return ValidationStatus.PRIVILEGE_REQUIRED

        if not analysis['executable']:
            return ValidationStatus.INVALID

        if analysis['errors']:
            # Check if errors are repairable
            if analysis['requires_privilege'] or 'option' in str(analysis['errors']).lower():
                return ValidationStatus.REPAIRABLE
            return ValidationStatus.INVALID

        return ValidationStatus.VALID

    def _build_result(
            self,
            status: ValidationStatus,
            command: str,
            execution_time: float = 0.0,
            exit_code: int = -1,
            stdout: str = '',
            stderr: str = '',
            errors: List[str] = None,
            warnings: List[str] = None,
            suggestions: List[str] = None,
            metadata: Dict = None
    ) -> ValidationResult:
        """
        Build structured validation result.

        Args:
            status: Validation status
            command: Original command
            execution_time: Execution duration
            exit_code: Process exit code
            stdout: Standard output
            stderr: Standard error
            errors: List of errors
            warnings: List of warnings
            suggestions: List of suggestions
            metadata: Additional metadata

        Returns:
            ValidationResult object
        """
        return ValidationResult(
            status=status.value,
            command=command,
            is_valid=status == ValidationStatus.VALID,
            is_safe=status != ValidationStatus.UNSAFE,
            requires_privilege=status == ValidationStatus.PRIVILEGE_REQUIRED,
            execution_time=execution_time,
            exit_code=exit_code,
            stdout=stdout[:500],  # Truncate for brevity
            stderr=stderr[:500],
            errors=errors or [],
            warnings=warnings or [],
            suggestions=suggestions or [],
            metadata=metadata or {}
        )

    def validate_to_json(self, command: str) -> str:
        """
        Validate and return JSON formatted result.

        Args:
            command: Nmap command to validate

        Returns:
            JSON string of validation result
        """
        result = self.validate(command)
        return json.dumps(asdict(result), indent=2)


def main():
    """Example usage and testing"""

    validator = NmapValidator(timeout=30)

    # Test cases
    test_commands = [
        # Valid commands
        "nmap -sV -p 80,443 scanme.nmap.org",
        "nmap -sn 192.168.1.0/24",
        "nmap -A -T4 example.com",

        # Invalid commands
        "nmap --invalid-option scanme.nmap.org",
        "nmap -sU -sT scanme.nmap.org",  # Conflicting scan types

        # Privilege required
        "nmap -sU -p 53 scanme.nmap.org",
        "nmap -O scanme.nmap.org",

        # Unsafe
        "nmap --script exploit scanme.nmap.org",
        "nmap -sn 0.0.0.0/0",
    ]

    print("=" * 80)
    print("NMAP COMMAND VALIDATION AGENT - TEST RESULTS")
    print("=" * 80)

    for cmd in test_commands:
        print(f"\n{'─' * 80}")
        print(f"Command: {cmd}")
        print(f"{'─' * 80}")

        result = validator.validate(cmd)

        print(f"Status: {result.status.upper()}")
        print(f"Valid: {result.is_valid}")
        print(f"Safe: {result.is_safe}")
        print(f"Requires Privilege: {result.requires_privilege}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Exit Code: {result.exit_code}")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  ❌ {error}")

        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

        if result.suggestions:
            print(f"\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  💡 {suggestion}")

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()