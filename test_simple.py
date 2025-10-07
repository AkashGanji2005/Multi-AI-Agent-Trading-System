#!/usr/bin/env python3
"""
Simple Test Script for AI Negotiator
Tests core functionality with minimal dependencies
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        import torch
        print("✅ Core dependencies available")
        
        from src.environment.marketplace import ResourceType, AgentType, Resource, Trade
        print("✅ Core data structures imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_structures():
    """Test basic data structures"""
    print("\n📊 Testing data structures...")
    
    try:
        from src.environment.marketplace import ResourceType, AgentType, Resource, Trade
        
        # Test Resource creation
        resource = Resource(
            id="test_resource_1",
            type=ResourceType.ENERGY,
            quantity=100.0,
            quality=0.8
        )
        print(f"✅ Created resource: {resource.type.value} (qty: {resource.quantity})")
        
        # Test Trade creation
        trade = Trade(
            id="test_trade_1",
            buyer="buyer_1",
            seller="seller_1",
            resource=resource,
            agreed_price=150.0,
            timestamp=time.time()
        )
        print(f"✅ Created trade: {trade.buyer} -> {trade.seller} for ${trade.agreed_price}")
        
        return True
    except Exception as e:
        print(f"❌ Data structure test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agents():
    """Test agent creation without environment"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from src.agents.buyer_agent import BuyerAgent
        from src.agents.seller_agent import SellerAgent
        from src.environment.marketplace import AgentType
        
        # Create a buyer agent
        buyer = BuyerAgent(
            agent_id="test_buyer_1",
            initial_cash=1000.0,
            risk_tolerance=0.5,
            negotiation_patience=10
        )
        print(f"✅ Created buyer agent: {buyer.agent_id}")
        
        # Create a seller agent  
        seller = SellerAgent(
            agent_id="test_seller_1",
            initial_cash=1000.0,
            pricing_strategy="competitive",
            quality_focus=0.7
        )
        print(f"✅ Created seller agent: {seller.agent_id}")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_communication():
    """Test communication protocol"""
    print("\n💬 Testing communication...")
    
    try:
        from src.communication.protocol import CommunicationProtocol, MessageType, Channel
        
        # Create protocol
        comm = CommunicationProtocol()
        print("✅ Communication protocol created")
        
        # Test message creation (without sending)
        message_data = {
            "resource_type": "energy",
            "quantity": 50,
            "price": 100
        }
        print(f"✅ Message data prepared: {message_data}")
        
        return True
    except Exception as e:
        print(f"❌ Communication test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_system():
    """Test reward calculation"""
    print("\n🎯 Testing reward system...")
    
    try:
        from src.environment.reward_system import RewardCalculator
        from src.environment.marketplace import ResourceType, Resource, Trade
        
        # Create reward calculator
        calculator = RewardCalculator()
        print("✅ Reward calculator created")
        
        # Test trade reward calculation
        resource = Resource("r1", ResourceType.ENERGY, 50.0, 0.8)
        trade = Trade("t1", "buyer1", "seller1", resource, 100.0, time.time())
        trade.status = "completed"
        
        # Calculate rewards
        buyer_reward = calculator.calculate_trade_reward("buyer1", trade, 95.0)  # market price
        seller_reward = calculator.calculate_trade_reward("seller1", trade, 95.0)
        
        print(f"✅ Trade rewards calculated - Buyer: {buyer_reward:.2f}, Seller: {seller_reward:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Reward system test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_training_setup():
    """Test basic training components"""
    print("\n🎓 Testing training setup...")
    
    try:
        from src.training.marl_trainer import MARLTrainer
        
        # Just test import and basic initialization without full setup
        print("✅ Training components imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Training setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all simple tests"""
    print("🚀 AI NEGOTIATOR - SIMPLE FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Agent Creation", test_agents),
        ("Communication", test_communication),
        ("Reward System", test_reward_system),
        ("Training Setup", test_basic_training_setup)
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
    
    if passed_tests >= 4:  # Allow some failures for optional components
        print("🎉 Core functionality is working!")
        print("\n🚀 Next steps to try:")
        print("1. Run the dashboard: python -m streamlit run dashboard.py")
        print("2. Try a simple simulation: python run_simulation.py")
        print("3. Test main CLI: python main.py scenarios")
    else:
        print(f"⚠️ Several core tests failed. Check the errors above.")
    
    return passed_tests >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)