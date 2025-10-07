#!/usr/bin/env python3
"""
Basic Example: AI Negotiator Marketplace
Demonstrates core functionality with a simple setup
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.environment.marketplace import MarketplaceEnv, ResourceType, AgentType
from src.agents.agent_factory import AgentFactory
from src.communication.protocol import CommunicationManager
from src.environment.reward_system import RewardSystem

def main():
    print("🤝 AI Negotiator - Basic Example")
    print("=" * 60)
    
    # 1. Create a simple marketplace environment
    print("1. Creating marketplace environment...")
    env = MarketplaceEnv(
        num_agents=6,
        max_steps=100,
        resource_types=[ResourceType.ENERGY, ResourceType.DATA],
        communication_enabled=True,
        alliance_enabled=True,
        regulation_enabled=True
    )
    
    # 2. Create diverse agents
    print("2. Creating agents...")
    agents = AgentFactory.create_balanced_population(
        total_agents=6,
        custom_distribution={
            AgentType.BUYER: 0.33,
            AgentType.SELLER: 0.33,
            AgentType.MEDIATOR: 0.17,
            AgentType.SPECULATOR: 0.17
        }
    )
    
    print(f"   Created {len(agents)} agents:")
    for agent in agents:
        print(f"   - {agent.agent_id}: {agent.agent_type.value}")
        print(f"     Cash: ${agent.cash:.0f}, Portfolio: {dict(agent.portfolio)}")
    
    # 3. Initialize communication system
    print("\n3. Setting up communication...")
    comm_manager = CommunicationManager()
    comm_manager.start()
    
    for agent in agents:
        comm_manager.register_agent(agent.agent_id, agent.agent_type)
    
    # 4. Initialize reward system
    reward_system = RewardSystem()
    
    # 5. Run simulation
    print("\n4. Running simulation...")
    print("=" * 60)
    
    obs, info = env.reset()
    step_count = 0
    total_rewards = {agent.agent_id: 0.0 for agent in agents}
    
    # Simulation loop
    while step_count < 100:
        actions = {}
        
        # Get actions from all agents
        for agent in agents:
            if agent.agent_id in obs:
                try:
                    action = agent.get_action(obs[agent.agent_id])
                    actions[agent.agent_id] = action
                except Exception:
                    # Fallback to no-action
                    actions[agent.agent_id] = {
                        'trade_resource_type': 0,
                        'trade_quantity': [0],
                        'trade_price': [0],
                        'trade_target': 0,
                        'trade_action_type': 2,  # No action
                        'comm_enabled': 0,
                        'comm_message_type': 0,
                        'comm_target': 0,
                        'alliance_action': 0,
                        'alliance_target': 0
                    }
        
        # Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Calculate custom rewards using reward system
        for agent in agents:
            if agent.agent_id in rewards:
                # Use the sophisticated reward system
                custom_reward = reward_system.calculate_reward(
                    agent.agent_id,
                    agent.agent_type,
                    actions.get(agent.agent_id, {}),
                    obs.get(agent.agent_id, {}),
                    {
                        'current_market_prices': dict(env.market_prices),
                        'violations': env.violation_counts,
                        'market_stability': 0.8,
                        'recent_trades': {agent.agent_id: 0}
                    }
                )
                
                total_rewards[agent.agent_id] += custom_reward
                
                # Update agent experience
                agent.add_experience(
                    obs.get(agent.agent_id, {}),
                    actions.get(agent.agent_id, {}),
                    custom_reward,
                    obs.get(agent.agent_id, {}),
                    terminations.get(agent.agent_id, False),
                    infos.get(agent.agent_id, {})
                )
        
        step_count += 1
        
        # Print progress every 20 steps
        if step_count % 20 == 0:
            avg_reward = sum(total_rewards.values()) / len(total_rewards)
            completed_trades = len([t for t in env.trades.values() if t.status == 'completed'])
            pending_trades = len([t for t in env.trades.values() if t.status == 'pending'])
            
            print(f"Step {step_count:3d}: "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Trades: {completed_trades:2d} completed, {pending_trades:2d} pending | "
                  f"Alliances: {len(env.alliances):2d}")
            
            # Show market prices
            prices = {rt.value: f"${price:.1f}" for rt, price in env.market_prices.items()}
            print(f"           Market Prices: {prices}")
        
        # Check for early termination
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break
    
    # 6. Final Results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"Simulation completed in {step_count} steps")
    print(f"Average reward per agent: {sum(total_rewards.values()) / len(total_rewards):.2f}")
    
    # Market statistics
    completed_trades = [t for t in env.trades.values() if t.status == 'completed']
    failed_trades = [t for t in env.trades.values() if t.status == 'failed']
    
    print(f"\nMarket Activity:")
    print(f"  - Completed trades: {len(completed_trades)}")
    print(f"  - Failed trades: {len(failed_trades)}")
    print(f"  - Success rate: {len(completed_trades)/(len(completed_trades)+len(failed_trades))*100:.1f}%" if (completed_trades or failed_trades) else "  - No trades attempted")
    print(f"  - Active alliances: {len(env.alliances)}")
    print(f"  - Total violations: {sum(env.violation_counts.values())}")
    
    # Agent performance ranking
    print(f"\nAgent Performance Ranking:")
    sorted_agents = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
    
    for i, (agent_id, reward) in enumerate(sorted_agents):
        agent = next(a for a in agents if a.agent_id == agent_id)
        portfolio_value = sum(agent.portfolio.values())
        total_value = agent.cash + portfolio_value * 50  # Rough portfolio valuation
        
        print(f"  {i+1}. {agent_id:12} ({agent.agent_type.value:10}) | "
              f"Reward: {reward:7.2f} | "
              f"Cash: ${agent.cash:6.0f} | "
              f"Portfolio: {portfolio_value:5.1f} | "
              f"Total Value: ${total_value:6.0f}")
    
    # Show some completed trades
    if completed_trades:
        print(f"\nSample Completed Trades:")
        for i, trade in enumerate(completed_trades[:5]):  # Show first 5 trades
            print(f"  {i+1}. {trade.buyer} -> {trade.seller}: "
                  f"{trade.resource.quantity:.1f} {trade.resource.type.value} "
                  f"@ ${trade.agreed_price:.2f} "
                  f"(Quality: {trade.resource.quality:.2f})")
    
    # Show alliances
    if env.alliances:
        print(f"\nActive Alliances:")
        for alliance_id, alliance in env.alliances.items():
            members_str = ", ".join(alliance.members)
            print(f"  - {alliance_id}: {members_str} ({alliance.purpose})")
    
    # Communication statistics
    comm_stats = comm_manager.get_communication_stats()
    print(f"\nCommunication Statistics:")
    print(f"  - Total channels: {comm_stats['total_channels']}")
    print(f"  - Messages queued: {comm_stats['total_messages_queued']}")
    print(f"  - Active negotiations: {comm_stats['active_negotiations']}")
    print(f"  - Active auctions: {comm_stats['active_auctions']}")
    
    # Cleanup
    comm_manager.stop()
    
    print(f"\n🎉 Example completed successfully!")
    print(f"   Try running 'streamlit run dashboard.py' for an interactive experience!")

if __name__ == "__main__":
    main()