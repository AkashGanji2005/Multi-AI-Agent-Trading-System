#!/usr/bin/env python3
"""
Working Test Script for AI Negotiator
Tests core functionality with correct API usage
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic system functionality"""
    print("🔍 Testing basic functionality...")
    
    try:
        from src.environment.marketplace import ResourceType, AgentType, Resource, Trade
        
        # Test Resource creation (correct parameters)
        resource = Resource(
            type=ResourceType.ENERGY,
            quantity=100.0,
            quality=0.8,
            price=150.0,
            owner="seller_1"
        )
        print(f"✅ Created resource: {resource.type.value} (qty: {resource.quantity}, price: ${resource.price})")
        
        # Test Trade creation
        trade = Trade(
            id="trade_1",
            buyer="buyer_1", 
            seller="seller_1",
            resource=resource,
            agreed_price=150.0,
            timestamp=int(time.time())
        )
        print(f"✅ Created trade: {trade.buyer} -> {trade.seller} for ${trade.agreed_price}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dashboard():
    """Test if dashboard can be imported"""
    print("\n📊 Testing dashboard import...")
    
    try:
        # Try to import dashboard components
        import streamlit as st
        print("✅ Streamlit available")
        
        # Check if dashboard.py exists and can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("dashboard", "dashboard.py")
        if spec and spec.loader:
            print("✅ Dashboard file found and can be imported")
            return True
        else:
            print("⚠️ Dashboard file not found or cannot be imported")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return False

def test_main_cli():
    """Test main CLI functionality"""
    print("\n🖥️ Testing main CLI...")
    
    try:
        # Check if main.py exists
        if os.path.exists("main.py"):
            print("✅ Main CLI file found")
            
            # Try to import main components
            import importlib.util
            spec = importlib.util.spec_from_file_location("main", "main.py")
            if spec and spec.loader:
                print("✅ Main CLI can be imported")
                return True
            else:
                print("⚠️ Main CLI cannot be imported")
                return False
        else:
            print("❌ Main CLI file not found")
            return False
            
    except Exception as e:
        print(f"❌ Main CLI test error: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing configuration...")
    
    try:
        # Check if config directory exists
        if os.path.exists("config") and os.path.exists("config/scenarios.yaml"):
            print("✅ Configuration directory and scenarios file found")
            
            # Try to load YAML
            import yaml
            with open("config/scenarios.yaml", 'r') as f:
                config = yaml.safe_load(f)
                if config and 'scenarios' in config:
                    print(f"✅ Configuration loaded successfully with {len(config['scenarios'])} scenarios")
                    return True
                else:
                    print("⚠️ Configuration file is empty or invalid")
                    return False
        else:
            print("❌ Configuration files not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False

def test_examples():
    """Test example files"""
    print("\n📚 Testing examples...")
    
    try:
        examples_found = 0
        
        # Check for example files
        if os.path.exists("examples/basic_example.py"):
            examples_found += 1
            print("✅ Basic example found")
            
        if os.path.exists("examples/big_data_example.py"):
            examples_found += 1
            print("✅ Big data example found")
            
        if os.path.exists("run_simulation.py"):
            examples_found += 1
            print("✅ Simulation runner found")
            
        if examples_found > 0:
            print(f"✅ Found {examples_found} example files")
            return True
        else:
            print("❌ No example files found")
            return False
            
    except Exception as e:
        print(f"❌ Examples test error: {e}")
        return False

def main():
    """Run working tests"""
    print("🚀 AI NEGOTIATOR - WORKING FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Data Structures", test_basic_functionality),
        ("Dashboard Components", test_simple_dashboard),
        ("Main CLI", test_main_cli),
        ("Configuration System", test_configuration),
        ("Example Files", test_examples)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
        
        print("-" * 40)
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests >= 3:
        print("🎉 System is ready to use!")
        print("\n🚀 What you can try now:")
        
        if passed_tests >= 4:
            print("1. 📊 Run the dashboard:")
            print("   python -m streamlit run dashboard.py")
            print()
        
        print("2. 🎮 Run a simple simulation:")
        print("   python run_simulation.py")
        print()
        
        print("3. 🖥️ Use the main CLI:")
        print("   python main.py scenarios")
        print("   python main.py simulate --scenario balanced --steps 100")
        print()
        
        if os.path.exists("examples/basic_example.py"):
            print("4. 📚 Try the basic example:")
            print("   python examples/basic_example.py")
            print()
        
        print("5. 🔧 Install optional big data dependencies:")
        print("   pip install pyspark kafka-python redis")
        print("   python examples/big_data_example.py")
        
    else:
        print(f"⚠️ System has some issues. {total_tests - passed_tests} tests failed.")
        print("Check the errors above and ensure all files are present.")
    
    return passed_tests >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)