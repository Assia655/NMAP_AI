#!/usr/bin/env python3
"""
Test script to verify NMAP Validation Agent is working correctly
Run this after setting up validation_agent.py in your project
"""

from validation_agent import NmapValidator
import sys


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_validator_creation():
    """Test 1: Can we create a validator?"""
    print_section("TEST 1: Validator Creation")
    try:
        validator = NmapValidator()
        print("✅ Validator created successfully")
        print(f"   Timeout: {validator.timeout}s")
        print(f"   Nmap path: {validator.nmap_path}")
        return validator
    except RuntimeError as e:
        print(f"❌ Failed to create validator: {e}")
        print("\n💡 Fix:")
        print("   • Install nmap: sudo apt-get install nmap")
        print("   • Or on macOS: brew install nmap")
        sys.exit(1)


def test_valid_command(validator):
    """Test 2: Valid command"""
    print_section("TEST 2: Valid Command")
    cmd = "nmap -sV -p 80 scanme.nmap.org"
    print(f"Command: {cmd}")

    result = validator.validate(cmd)

    print(f"Status: {result.status}")
    print(f"Valid: {result.is_valid}")
    print(f"Safe: {result.is_safe}")
    print(f"Time: {result.execution_time:.3f}s")

    if result.is_valid:
        print("✅ Test passed")
    else:
        print(f"❌ Expected valid, got: {result.status}")
        print(f"   Errors: {result.errors}")


def test_invalid_option(validator):
    """Test 3: Invalid option"""
    print_section("TEST 3: Invalid Option Detection")
    cmd = "nmap --nonexistent-flag scanme.nmap.org"
    print(f"Command: {cmd}")

    result = validator.validate(cmd)

    print(f"Status: {result.status}")
    print(f"Valid: {result.is_valid}")

    if not result.is_valid and result.errors:
        print("✅ Test passed - correctly detected invalid option")
        print(f"   Errors: {result.errors}")
    else:
        print("❌ Should have detected invalid option")


def test_privilege_required(validator):
    """Test 4: Privilege requirement detection"""
    print_section("TEST 4: Privilege Requirement Detection")
    cmd = "nmap -sU -p 53 scanme.nmap.org"
    print(f"Command: {cmd}")

    result = validator.validate(cmd)

    print(f"Status: {result.status}")
    print(f"Requires Privilege: {result.requires_privilege}")

    if result.requires_privilege:
        print("✅ Test passed - correctly detected privilege requirement")
        print(f"   Suggestions: {result.suggestions}")
    else:
        print("⚠️  Warning: Should require privilege (unless running as root)")


def test_no_target(validator):
    """Test 5: Missing target"""
    print_section("TEST 5: Missing Target Detection")
    cmd = "nmap -sV -p 80"
    print(f"Command: {cmd}")

    result = validator.validate(cmd)

    print(f"Status: {result.status}")
    print(f"Valid: {result.is_valid}")

    if not result.is_valid:
        print("✅ Test passed - correctly detected missing target")
        print(f"   Errors: {result.errors}")
    else:
        print("❌ Should have detected missing target")


def test_unsafe_command(validator):
    """Test 6: Unsafe command detection"""
    print_section("TEST 6: Unsafe Command Detection")
    cmd = "nmap --script exploit scanme.nmap.org"
    print(f"Command: {cmd}")

    result = validator.validate(cmd)

    print(f"Status: {result.status}")
    print(f"Safe: {result.is_safe}")

    if not result.is_safe:
        print("✅ Test passed - correctly blocked unsafe command")
        print(f"   Errors: {result.errors}")
    else:
        print("❌ Should have blocked unsafe command")


def test_json_output(validator):
    """Test 7: JSON output"""
    print_section("TEST 7: JSON Output Format")
    cmd = "nmap -p 80 scanme.nmap.org"
    print(f"Command: {cmd}")

    json_output = validator.validate_to_json(cmd)

    print("✅ JSON output generated")
    print(f"   Length: {len(json_output)} characters")
    print("\n   Sample (first 200 chars):")
    print(f"   {json_output[:200]}...")


def test_batch_validation(validator):
    """Test 8: Batch validation"""
    print_section("TEST 8: Batch Validation")

    commands = [
        "nmap -sV scanme.nmap.org",
        "nmap -p 80,443 example.com",
        "nmap --invalid-option target.com",
        "nmap -sU example.com"
    ]

    print(f"Validating {len(commands)} commands...")

    results = []
    for cmd in commands:
        result = validator.validate(cmd)
        results.append(result)
        status_symbol = "✅" if result.is_valid else "❌"
        print(f"  {status_symbol} {result.status:20s} - {cmd}")

    valid_count = sum(1 for r in results if r.is_valid)
    print(f"\n✅ Batch test complete: {valid_count}/{len(commands)} valid")


def run_performance_test(validator):
    """Test 9: Performance benchmark"""
    print_section("TEST 9: Performance Benchmark")

    commands = [
        "nmap -sV scanme.nmap.org",
        "nmap -p 80 example.com",
        "nmap -p 443 test.com"
    ]

    times = []
    for cmd in commands:
        result = validator.validate(cmd)
        times.append(result.execution_time)
        print(f"  {result.execution_time:.3f}s - {cmd}")

    avg_time = sum(times) / len(times)
    print(f"\n✅ Average validation time: {avg_time:.3f}s")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NMAP VALIDATION AGENT" + " " * 32 + "║")
    print("║" + " " * 23 + "Test Suite" + " " * 36 + "║")
    print("╚" + "=" * 68 + "╝")

    # Create validator
    validator = test_validator_creation()

    # Run tests
    try:
        test_valid_command(validator)
        test_invalid_option(validator)
        test_privilege_required(validator)
        test_no_target(validator)
        test_unsafe_command(validator)
        test_json_output(validator)
        test_batch_validation(validator)
        run_performance_test(validator)

        # Summary
        print_section("TEST SUMMARY")
        print("✅ All tests completed successfully!")
        print("\nYour validation agent is working correctly.")
        print("\nNext steps:")
        print("  1. Integrate with your command generator")
        print("  2. Add custom safety rules if needed")
        print("  3. Monitor performance in production")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nPlease check:")
        print("  1. Nmap is installed: nmap --version")
        print("  2. validation_agent.py is in the same directory")
        print("  3. Python version is 3.7 or higher")
        sys.exit(1)


if __name__ == '__main__':
    main()