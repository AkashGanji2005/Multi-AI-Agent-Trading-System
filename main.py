#!/usr/bin/env python3
"""
AI Negotiator: Multi-Agent Reinforcement Learning Marketplace
Main entry point for training and running the marketplace simulation
"""

import argparse
import os
import sys
import yaml
import logging
from typing import Dict, Any
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.marketplace import MarketplaceEnv
from src.agents.agent_factory import AgentFactory
from src.training.marl_trainer import MARLTrainer, TrainingConfig, create_training_config
from src.communication.protocol import CommunicationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """Load scenario configuration from YAML file"""
    
    config_path = os.path.join("config", "scenarios.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        if scenario_name not in all_configs:
            logger.error(f"Scenario '{scenario_name}' not found in configuration")
            available = list(all_configs.keys())
            logger.info(f"Available scenarios: {available}")
            return {}
        
        return all_configs[scenario_name]
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def run_simulation(args):
    """Run a marketplace simulation"""
    
    logger.info(f"Starting marketplace simulation with scenario: {args.scenario}")
    
    # Load scenario configuration
    scenario_config = load_scenario_config(args.scenario)
    if not scenario_config:
        logger.error("Failed to load scenario configuration")
        return
    
    env_config = scenario_config.get('environment', {})
    agent_config = scenario_config.get('agents', {})
    
    # Create environment
    env = MarketplaceEnv(
        num_agents=env_config.get('num_agents', 15),
        max_steps=env_config.get('max_steps', 1000),
        resource_types=env_config.get('resource_types', ['energy', 'data', 'goods', 'services']),
        initial_resources=env_config.get('initial_resources', {}),
        communication_enabled=env_config.get('communication_enabled', True),
        alliance_enabled=env_config.get('alliance_enabled', True),
        regulation_enabled=env_config.get('regulation_enabled', True)
    )
    
    # Create agents
    distribution = agent_config.get('distribution', {})
    total_agents = env_config.get('num_agents', 15)
    
    agents = AgentFactory.create_balanced_population(
        total_agents=total_agents,
        custom_distribution=distribution
    )
    
    # Initialize communication manager
    comm_manager = CommunicationManager()
    comm_manager.start()
    
    # Register agents
    for agent in agents:
        comm_manager.register_agent(agent.agent_id, agent.agent_type)
    
    logger.info(f"Created {len(agents)} agents")
    
    # Run simulation
    try:
        obs, info = env.reset()
        total_rewards = {agent.agent_id: 0.0 for agent in agents}
        step_count = 0
        
        print(f"\n{'='*60}")
        print(f"MARKETPLACE SIMULATION - {scenario_config.get('name', 'Unknown Scenario')}")
        print(f"{'='*60}")
        print(f"Agents: {len(agents)} | Max Steps: {env_config.get('max_steps', 1000)}")
        print(f"{'='*60}\n")
        
        while step_count < args.max_steps:
            actions = {}
            
            # Get actions from all agents
            for agent in agents:
                if agent.agent_id in obs:
                    action = agent.get_action(obs[agent.agent_id])
                    actions[agent.agent_id] = action
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update agent experiences
            for agent in agents:
                if agent.agent_id in rewards:
                    total_rewards[agent.agent_id] += rewards[agent.agent_id]
                    
                    # Simple experience storage (for potential learning)
                    agent.add_experience(
                        obs.get(agent.agent_id, {}),
                        actions.get(agent.agent_id, {}),
                        rewards[agent.agent_id],
                        obs.get(agent.agent_id, {}),
                        terminations.get(agent.agent_id, False),
                        infos.get(agent.agent_id, {})
                    )
            
            step_count += 1
            
            # Print progress
            if step_count % 100 == 0:
                avg_reward = sum(total_rewards.values()) / len(total_rewards)
                print(f"Step {step_count}: Average Reward = {avg_reward:.2f}")
                
                # Show market state
                if hasattr(env, 'market_prices'):
                    prices = {rt.value: price for rt, price in env.market_prices.items()}
                    print(f"Market Prices: {prices}")
                
                print(f"Active Trades: {len([t for t in env.trades.values() if t.status == 'pending'])}")
                print(f"Alliances: {len(env.alliances)}")
                print("-" * 40)
            
            # Check for early termination
            if terminations.get("__all__", False) or truncations.get("__all__", False):
                break
        
        # Final results
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Steps: {step_count}")
        print(f"Average Reward: {sum(total_rewards.values()) / len(total_rewards):.2f}")
        print(f"Total Trades Completed: {len([t for t in env.trades.values() if t.status == 'completed'])}")
        print(f"Final Alliances: {len(env.alliances)}")
        
        # Agent performance summary
        print(f"\nTOP PERFORMERS:")
        sorted_agents = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
        for i, (agent_id, reward) in enumerate(sorted_agents[:5]):
            agent_type = next(a.agent_type.value for a in agents if a.agent_id == agent_id)
            print(f"{i+1}. {agent_id} ({agent_type}): {reward:.2f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise
    finally:
        comm_manager.stop()
        print("\nSimulation completed")

def run_training(args):
    """Run MARL training"""
    
    logger.info(f"Starting MARL training with scenario: {args.scenario}")
    
    # Load scenario configuration
    scenario_config = load_scenario_config(args.scenario)
    if not scenario_config:
        logger.error("Failed to load scenario configuration")
        return
    
    training_config = scenario_config.get('training', {})
    env_config = scenario_config.get('environment', {})
    
    # Create training configuration
    config = create_training_config(
        scenario=args.scenario,
        algorithm=training_config.get('algorithm', 'PPO'),
        num_agents=env_config.get('num_agents', 15),
        max_episode_steps=env_config.get('max_steps', 1000),
        learning_rate=training_config.get('learning_rate', 3e-4),
        num_iterations=args.iterations or training_config.get('num_iterations', 1000),
        num_workers=args.workers or training_config.get('num_workers', 4),
        batch_size=training_config.get('batch_size', 512),
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        curriculum_enabled=training_config.get('curriculum_enabled', False)
    )
    
    print(f"\n{'='*60}")
    print(f"MARL TRAINING - {scenario_config.get('name', 'Unknown Scenario')}")
    print(f"{'='*60}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Agents: {config.num_agents}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Workers: {config.num_workers}")
    print(f"{'='*60}\n")
    
    # Create and run trainer
    trainer = MARLTrainer(config)
    
    try:
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
        
        # Run training
        start_time = time.time()
        metrics = trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Training Time: {training_time/3600:.1f} hours")
        print(f"Final Average Reward: {metrics['episode_rewards'][-1]:.2f}")
        print(f"Best Reward: {max(metrics['episode_rewards']):.2f}")
        
        # Save final model
        if args.save_model:
            model_path = trainer.save_model(args.save_model)
            print(f"Model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        trainer.close()

def run_evaluation(args):
    """Run model evaluation"""
    
    logger.info(f"Starting model evaluation")
    
    if not args.checkpoint:
        logger.error("Checkpoint path required for evaluation")
        return
    
    # Load scenario configuration
    scenario_config = load_scenario_config(args.scenario)
    if not scenario_config:
        logger.error("Failed to load scenario configuration")
        return
    
    training_config = scenario_config.get('training', {})
    env_config = scenario_config.get('environment', {})
    
    # Create training configuration
    config = create_training_config(
        scenario=args.scenario,
        algorithm=training_config.get('algorithm', 'PPO'),
        num_agents=env_config.get('num_agents', 15),
        max_episode_steps=env_config.get('max_steps', 1000)
    )
    
    # Create trainer and load model
    trainer = MARLTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION - {scenario_config.get('name', 'Unknown Scenario')}")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"{'='*60}\n")
    
    try:
        # Run evaluation
        results = trainer.evaluate(num_episodes=args.eval_episodes)
        
        print("EVALUATION RESULTS:")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.1f}")
        print(f"Market Efficiency: {results['mean_market_efficiency']:.3f}")
        print(f"Cooperation Score: {results['mean_cooperation_score']:.3f}")
        print(f"Violation Rate: {results['mean_violation_rate']:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise
    finally:
        trainer.close()

def list_scenarios():
    """List available scenarios"""
    
    config_path = os.path.join("config", "scenarios.yaml")
    
    if not os.path.exists(config_path):
        print("Configuration file not found")
        return
    
    try:
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        print(f"\n{'='*60}")
        print("AVAILABLE SCENARIOS")
        print(f"{'='*60}")
        
        for name, config in all_configs.items():
            print(f"\n{name.upper()}:")
            print(f"  Name: {config.get('name', 'N/A')}")
            print(f"  Description: {config.get('description', 'N/A')}")
            
            env_config = config.get('environment', {})
            print(f"  Agents: {env_config.get('num_agents', 'N/A')}")
            print(f"  Max Steps: {env_config.get('max_steps', 'N/A')}")
            print(f"  Resources: {env_config.get('resource_types', [])}")
        
        print(f"\n{'='*60}")
        
    except Exception as e:
        logger.error(f"Error loading scenarios: {e}")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="AI Negotiator: Multi-Agent Reinforcement Learning Marketplace"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run marketplace simulation')
    sim_parser.add_argument('--scenario', default='balanced', help='Scenario to run')
    sim_parser.add_argument('--max-steps', type=int, default=1000, help='Maximum simulation steps')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train MARL agents')
    train_parser.add_argument('--scenario', default='balanced', help='Training scenario')
    train_parser.add_argument('--iterations', type=int, help='Training iterations')
    train_parser.add_argument('--workers', type=int, help='Number of workers')
    train_parser.add_argument('--checkpoint', help='Load from checkpoint')
    train_parser.add_argument('--save-model', help='Save trained model path')
    train_parser.add_argument('--wandb-project', help='Weights & Biases project name')
    train_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--scenario', default='balanced', help='Evaluation scenario')
    eval_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    eval_parser.add_argument('--eval-episodes', type=int, default=20, help='Evaluation episodes')
    
    # List scenarios command
    subparsers.add_parser('scenarios', help='List available scenarios')
    
    args = parser.parse_args()
    
    if args.command == 'simulate':
        run_simulation(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'scenarios':
        list_scenarios()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()