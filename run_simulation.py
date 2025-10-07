#!/usr/bin/env python3
"""
Quick simulation runner for testing
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.marketplace import MarketplaceEnv
from src.agents.agent_factory import create_test_population
from src.communication.protocol import CommunicationManager

def main():
    print("🤝 AI Negotiator - Quick Simulation")
    print("=" * 50)
    
    # Create test environment
    env = MarketplaceEnv(
        num_agents=8,
        max_steps=200,
        communication_enabled=True,
        alliance_enabled=True,
        regulation_enabled=True
    )
    
    # Create test agents
    agents = create_test_population()
    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.agent_id} ({agent.agent_type.value})")
    
    # Initialize communication
    comm_manager = CommunicationManager()
    comm_manager.start()
    
    # Register agents
    for agent in agents:
        comm_manager.register_agent(agent.agent_id, agent.agent_type)
    
    print("\nStarting simulation...")
    print("=" * 50)
    
    try:
        # Reset environment
        obs, info = env.reset()
        total_rewards = {agent.agent_id: 0.0 for agent in agents}
        
        for step in range(200):
            actions = {}
            
            # Get actions from agents
            for agent in agents:
                if agent.agent_id in obs:
                    try:
                        action = agent.get_action(obs[agent.agent_id])
                        actions[agent.agent_id] = action
                    except:
                        # Fallback action
                        actions[agent.agent_id] = {
                            'trade_resource_type': 0,
                            'trade_quantity': [0],
                            'trade_price': [0],
                            'trade_target': 0,
                            'trade_action_type': 2,
                            'comm_enabled': 0,
                            'comm_message_type': 0,
                            'comm_target': 0,
                            'alliance_action': 0,
                            'alliance_target': 0
                        }
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update rewards
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            # Print progress
            if step % 50 == 0:
                avg_reward = sum(total_rewards.values()) / len(total_rewards)
                completed_trades = len([t for t in env.trades.values() if t.status == 'completed'])
                print(f"Step {step}: Avg Reward = {avg_reward:.2f}, Trades = {completed_trades}")
            
            # Check termination
            if terminations.get("__all__", False) or truncations.get("__all__", False):
                break
        
        # Final results
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETED")
        print("=" * 50)
        
        print(f"Total Steps: {step + 1}")
        print(f"Average Reward: {sum(total_rewards.values()) / len(total_rewards):.2f}")
        print(f"Completed Trades: {len([t for t in env.trades.values() if t.status == 'completed'])}")
        print(f"Active Alliances: {len(env.alliances)}")
        
        # Top performers
        print("\nTop Performers:")
        sorted_rewards = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
        for i, (agent_id, reward) in enumerate(sorted_rewards[:3]):
            agent_type = next(a.agent_type.value for a in agents if a.agent_id == agent_id)
            print(f"{i+1}. {agent_id} ({agent_type}): {reward:.2f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        raise
    finally:
        comm_manager.stop()
        print("\nSimulation finished!")

if __name__ == "__main__":
    main()