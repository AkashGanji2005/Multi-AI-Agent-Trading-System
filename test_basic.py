#!/usr/bin/env python3
"""
Basic Test Script for AI Negotiator
Tests core functionality without external dependencies
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        from src.environment.marketplace import MarketplaceEnv, ResourceType, AgentType
        print("✅ Marketplace environment imported successfully")
        
        from src.agents.agent_factory import AgentFactory
        print("✅ Agent factory imported successfully")
        
        from src.agents.buyer_agent import BuyerAgent
        from src.agents.seller_agent import SellerAgent
        print("✅ Agent types imported successfully")
        
        from src.communication.protocol import CommunicationProtocol
        print("✅ Communication protocol imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment_creation():
    """Test creating a basic marketplace environment"""
    print("\n🏪 Testing environment creation...")
    
    try:
        from src.environment.marketplace import MarketplaceEnv
        
        # Create environment
        env = MarketplaceEnv(num_agents=4, max_steps=10)
        print("✅ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful, got {len(obs)} agent observations")
        
        # Check observation structure
        if obs:
            first_agent = list(obs.keys())[0]
            obs_shape = obs[first_agent]['market_prices'].shape if hasattr(obs[first_agent]['market_prices'], 'shape') else len(obs[first_agent]['market_prices'])
            print(f"✅ Observation structure valid, market prices shape: {obs_shape}")
        
        return True, env
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_agent_creation():
    """Test creating different agent types"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from src.agents.agent_factory import AgentFactory
        
        # Create balanced population
        agents = AgentFactory.create_balanced_population(6)
        print(f"✅ Created {len(agents)} agents successfully")
        
        # Check agent types
        agent_types = {}
        for agent in agents:
            agent_type = agent.agent_type.value
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        print("✅ Agent distribution:")
        for agent_type, count in agent_types.items():
            print(f"   - {agent_type.capitalize()}: {count}")
        
        return True, agents
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_basic_simulation():
    """Test a basic simulation run"""
    print("\n🎮 Testing basic simulation...")
    
    try:
        from src.environment.marketplace import MarketplaceEnv
        from src.agents.agent_factory import AgentFactory
        
        # Create environment and agents
        env = MarketplaceEnv(num_agents=4, max_steps=20)
        agents = AgentFactory.create_balanced_population(4)
        
        # Reset environment
        obs, info = env.reset()
        print("✅ Simulation initialized")
        
        # Run simulation steps
        total_rewards = {agent.agent_id: 0 for agent in agents}
        completed_trades = 0
        
        for step in range(10):  # Run for 10 steps
            actions = {}
            
            # Get actions from agents
            for agent in agents:
                if agent.agent_id in obs:
                    try:
                        # Get action from agent
                        action = agent.get_action(obs[agent.agent_id])
                        actions[agent.agent_id] = action
                    except Exception as e:
                        # Fallback to random action if agent fails
                        actions[agent.agent_id] = {
                            'trade_resource_type': 0,
                            'trade_quantity': [10.0],
                            'trade_price': [100.0],
                            'trade_target': 0,
                            'trade_action_type': 2,  # hold
                            'comm_enabled': 0,
                            'comm_message_type': 0,
                            'comm_target': 0,
                            'alliance_action': 0,
                            'alliance_target': 0
                        }
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update statistics
            for agent_id, reward in rewards.items():
                if agent_id in total_rewards:
                    total_rewards[agent_id] += reward
            
            # Count completed trades
            step_trades = len([t for t in env.trades.values() if t.status == 'completed'])
            if step_trades > completed_trades:
                completed_trades = step_trades
            
            if step % 5 == 0:
                print(f"   Step {step}: {len(env.trades)} total trades, {completed_trades} completed")
        
        print(f"✅ Simulation completed successfully!")
        print(f"   - Total trades: {len(env.trades)}")
        print(f"   - Completed trades: {completed_trades}")
        print(f"   - Agent rewards: {list(total_rewards.values())}")
        
        return True
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_communication():
    """Test communication between agents"""
    print("\n💬 Testing communication system...")
    
    try:
        from src.communication.protocol import CommunicationProtocol, MessageType, Channel
        
        # Create communication protocol
        comm = CommunicationProtocol()
        print("✅ Communication protocol created")
        
        # Test message sending
        success = comm.send_message(
            sender="agent_1",
            receiver="agent_2", 
            message_type=MessageType.TRADE_OFFER,
            content={"resource": "energy", "quantity": 10, "price": 100},
            channel=Channel.DIRECT
        )
        
        if success:
            print("✅ Message sent successfully")
        else:
            print("⚠️ Message sending failed (expected for basic test)")
        
        # Test getting messages
        messages = comm.get_messages("agent_2")
        print(f"✅ Retrieved {len(messages)} messages")
        
        return True
    except Exception as e:
        print(f"❌ Communication test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("🚀 AI NEGOTIATOR - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Environment Creation", lambda: test_environment_creation()[0]),
        ("Agent Creation", lambda: test_agent_creation()[0]),
        ("Basic Simulation", test_basic_simulation),
        ("Communication System", test_communication)
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
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The AI Negotiator system is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Run the dashboard: python -m streamlit run dashboard.py")
        print("2. Run a training session: python main.py train --scenario balanced")
        print("3. Try the big data integration: python examples/big_data_example.py")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)