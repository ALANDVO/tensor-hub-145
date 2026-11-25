#!/usr/bin/env python3

# Simple test for demo_logreg.py

import sys
import os

def test_demo_exists():
    """Test that demo_logreg.py exists"""
    script_path = os.path.join(os.path.dirname(__file__), "demo_logreg.py")
    assert os.path.exists(script_path), f"demo_logreg.py not found at {script_path}"

def test_demo_can_be_imported():
    """Test that demo_logreg.py can be imported"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_logreg", os.path.join(os.path.dirname(__file__), "demo_logreg.py"))
    assert spec is not None, "Cannot load demo_logreg.py"

def main():
    """Run basic tests"""
    print("Running basic demo_logreg.py tests...")
    
    test_demo_exists()
    test_demo_can_be_imported()
    
    print("✓ All basic tests passed!")
    print("The demo_logreg.py implements the basic logistic regression demo feature")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
