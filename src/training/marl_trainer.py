"""
Multi-Agent Reinforcement Learning Training Pipeline
Supports multiple MARL algorithms and training strategies
"""

import os
import json
import time
import numpy as np
import torch
import wandb
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
import gymnasium as gym

from ..environment.marketplace import MarketplaceEnv
from ..agents.agent_factory import AgentFactory, create_test_population
from ..agents.base_agent import MarketplaceFeatureExtractor
from ..communication.protocol import CommunicationManager

torch, nn = try_import_torch()

@dataclass
class TrainingConfig:
    """Configuration for MARL training"""
    
    # Algorithm settings
    algorithm: str = "PPO"  # PPO, DQN, SAC, MADDPG
    framework: str = "torch"
    
    # Environment settings
    num_agents: int = 15
    max_episode_steps: int = 1000
    scenario: str = "balanced"
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 512
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    vf_loss_coeff: float = 0.5
    
    # Multi-agent settings
    policies_to_train: List[str] = None
    policy_mapping_fn: Optional[Callable] = None
    shared_policy: bool = False
    
    # Training schedule
    num_iterations: int = 1000
    checkpoint_freq: int = 50
    evaluation_freq: int = 10
    evaluation_episodes: int = 10
    
    # Resource allocation
    num_workers: int = 4
    num_gpus: int = 0
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: float = 0.0
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[Dict[str, Any]] = None
    
    # Logging and monitoring
    log_level: str = "INFO"
    wandb_project: str = "ai-negotiator"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        if self.policies_to_train is None:
            self.policies_to_train = ["shared_policy"]
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                {"name": "basic", "max_agents": 5, "max_steps": 200, "iterations": 100},
                {"name": "intermediate", "max_agents": 10, "max_steps": 500, "iterations": 300},
                {"name": "advanced", "max_agents": 15, "max_steps": 1000, "iterations": 600}
            ]

class CustomMarketplaceModel(TorchModelV2, nn.Module):
    """Custom neural network model for marketplace agents"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        
        # Use the marketplace feature extractor
        self.feature_extractor = MarketplaceFeatureExtractor(obs_space, features_dim=512)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self._value_out = None
    
    def forward(self, input_dict, state, seq_lens):
        features = self.feature_extractor(input_dict["obs"])
        policy_out = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)
        return policy_out, state
    
    def value_function(self):
        return self._value_out

class MARLTrainer:
    """
    Multi-Agent Reinforcement Learning trainer with support for various algorithms
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.algorithm = None
        self.env_creator = None
        self.communication_manager = CommunicationManager()
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.num_workers + 1,
                num_gpus=config.num_gpus,
                log_to_driver=True
            )
        
        # Register custom model
        ModelCatalog.register_custom_model("marketplace_model", CustomMarketplaceModel)
        
        # Setup logging
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                name=f"marl_training_{int(time.time())}"
            )
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'market_efficiency': [],
            'cooperation_scores': [],
            'violation_rates': []
        }
        
        self.current_curriculum_stage = 0
    
    def create_env(self, env_config: Dict[str, Any] = None):
        """Create marketplace environment"""
        
        env_config = env_config or {}
        
        # Get current curriculum stage if enabled
        if self.config.curriculum_enabled and self.current_curriculum_stage < len(self.config.curriculum_stages):
            stage = self.config.curriculum_stages[self.current_curriculum_stage]
            env_config.update({
                'num_agents': stage.get('max_agents', self.config.num_agents),
                'max_steps': stage.get('max_steps', self.config.max_episode_steps)
            })
        else:
            env_config.update({
                'num_agents': self.config.num_agents,
                'max_steps': self.config.max_episode_steps
            })
        
        # Create marketplace environment
        env = MarketplaceEnv(
            num_agents=env_config.get('num_agents', self.config.num_agents),
            max_steps=env_config.get('max_steps', self.config.max_episode_steps),
            communication_enabled=True,
            alliance_enabled=True,
            regulation_enabled=True
        )
        
        # Wrap for RLLib
        return PettingZooEnv(env)
    
    def setup_policies(self) -> Dict[str, PolicySpec]:
        """Setup multi-agent policies"""
        
        # Create sample environment to get spaces
        sample_env = self.create_env()
        
        policies = {}
        
        if self.config.shared_policy:
            # Single shared policy for all agents
            policies["shared_policy"] = PolicySpec(
                policy_class=None,  # Use default policy class
                observation_space=sample_env.observation_space,
                action_space=sample_env.action_space,
                config={
                    "model": {
                        "custom_model": "marketplace_model",
                        "custom_model_config": {}
                    }
                }
            )
        else:
            # Separate policies for each agent type
            agent_types = ["buyer", "seller", "regulator", "mediator", "speculator"]
            
            for agent_type in agent_types:
                policies[f"{agent_type}_policy"] = PolicySpec(
                    policy_class=None,
                    observation_space=sample_env.observation_space,
                    action_space=sample_env.action_space,
                    config={
                        "model": {
                            "custom_model": "marketplace_model",
                            "custom_model_config": {"agent_type": agent_type}
                        }
                    }
                )
        
        sample_env.close()
        return policies
    
    def policy_mapping_function(self, agent_id: str, episode: Any = None, worker: Any = None, **kwargs) -> str:
        """Map agents to policies"""
        
        if self.config.shared_policy:
            return "shared_policy"
        else:
            # Extract agent type from agent_id (format: "type_number")
            agent_type = agent_id.split('_')[0]
            return f"{agent_type}_policy"
    
    def setup_algorithm(self):
        """Setup the MARL algorithm"""
        
        # Register environment
        register_env("marketplace", self.create_env)
        
        # Setup policies
        policies = self.setup_policies()
        
        # Common configuration
        common_config = {
            "env": "marketplace",
            "framework": self.config.framework,
            "num_workers": self.config.num_workers,
            "num_gpus": self.config.num_gpus,
            "num_cpus_per_worker": self.config.num_cpus_per_worker,
            "num_gpus_per_worker": self.config.num_gpus_per_worker,
            "log_level": self.config.log_level,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": self.policy_mapping_function,
                "policies_to_train": self.config.policies_to_train
            },
            "model": {
                "custom_model": "marketplace_model"
            },
            "evaluation_config": {
                "explore": False,
                "env_config": {"evaluation": True}
            }
        }
        
        # Algorithm-specific configuration
        if self.config.algorithm.upper() == "PPO":
            config = PPOConfig()
            config.update_from_dict(common_config)
            config.update_from_dict({
                "lr": self.config.learning_rate,
                "train_batch_size": self.config.train_batch_size,
                "sgd_minibatch_size": self.config.sgd_minibatch_size,
                "num_sgd_iter": self.config.num_sgd_iter,
                "gamma": self.config.gamma,
                "lambda": self.config.lambda_,
                "clip_param": self.config.clip_param,
                "entropy_coeff": self.config.entropy_coeff,
                "vf_loss_coeff": self.config.vf_loss_coeff
            })
            
        elif self.config.algorithm.upper() == "DQN":
            config = DQNConfig()
            config.update_from_dict(common_config)
            config.update_from_dict({
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "target_network_update_freq": 500,
                "buffer_size": 50000,
                "learning_starts": 1000
            })
            
        elif self.config.algorithm.upper() == "SAC":
            config = SACConfig()
            config.update_from_dict(common_config)
            config.update_from_dict({
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "target_entropy": "auto",
                "tau": 0.005,
                "buffer_size": 50000
            })
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Build algorithm
        self.algorithm = config.build()
    
    def train(self):
        """Main training loop"""
        
        if self.algorithm is None:
            self.setup_algorithm()
        
        print(f"Starting MARL training with {self.config.algorithm}")
        print(f"Training for {self.config.num_iterations} iterations")
        
        best_reward = float('-inf')
        
        for iteration in range(self.config.num_iterations):
            # Check for curriculum progression
            if self.config.curriculum_enabled:
                self._check_curriculum_progression(iteration)
            
            # Training step
            start_time = time.time()
            result = self.algorithm.train()
            training_time = time.time() - start_time
            
            # Extract metrics
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            
            # Custom metrics
            custom_metrics = result.get("custom_metrics", {})
            market_efficiency = custom_metrics.get("market_efficiency", 0)
            cooperation_score = custom_metrics.get("cooperation_score", 0)
            violation_rate = custom_metrics.get("violation_rate", 0)
            
            # Store metrics
            self.training_metrics['episode_rewards'].append(episode_reward_mean)
            self.training_metrics['episode_lengths'].append(episode_len_mean)
            self.training_metrics['market_efficiency'].append(market_efficiency)
            self.training_metrics['cooperation_scores'].append(cooperation_score)
            self.training_metrics['violation_rates'].append(violation_rate)
            
            # Logging
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                      f"Reward={episode_reward_mean:.2f}, "
                      f"Length={episode_len_mean:.1f}, "
                      f"Efficiency={market_efficiency:.3f}, "
                      f"Time={training_time:.1f}s")
            
            # Wandb logging
            if self.config.wandb_project:
                wandb.log({
                    "iteration": iteration,
                    "episode_reward_mean": episode_reward_mean,
                    "episode_len_mean": episode_len_mean,
                    "market_efficiency": market_efficiency,
                    "cooperation_score": cooperation_score,
                    "violation_rate": violation_rate,
                    "training_time": training_time
                })
            
            # Evaluation
            if iteration % self.config.evaluation_freq == 0 and iteration > 0:
                eval_results = self.evaluate()
                print(f"Evaluation at iteration {iteration}: {eval_results}")
                
                if self.config.wandb_project:
                    wandb.log({"eval_" + k: v for k, v in eval_results.items()})
            
            # Checkpointing
            if (iteration % self.config.checkpoint_freq == 0 and iteration > 0) or episode_reward_mean > best_reward:
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                
                if self.config.save_checkpoints:
                    checkpoint_path = self.algorithm.save(self.config.checkpoint_dir)
                    print(f"Checkpoint saved at: {checkpoint_path}")
                    
                    # Save training metrics
                    metrics_path = os.path.join(self.config.checkpoint_dir, "training_metrics.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(self.training_metrics, f, indent=2)
        
        print("Training completed!")
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=self.config.evaluation_episodes * 2)
        print(f"Final evaluation: {final_eval}")
        
        return self.training_metrics
    
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate the trained policies"""
        
        num_episodes = num_episodes or self.config.evaluation_episodes
        
        # Create evaluation environment
        eval_env = self.create_env({"evaluation": True})
        
        episode_rewards = []
        episode_lengths = []
        market_efficiencies = []
        cooperation_scores = []
        violation_rates = []
        
        for episode in range(num_episodes):
            obs = eval_env.reset()
            done = {"__all__": False}
            episode_reward = 0
            episode_length = 0
            
            while not done["__all__"]:
                # Get actions from policies
                actions = {}
                for agent_id in obs.keys():
                    if agent_id != "__all__":
                        policy_id = self.policy_mapping_function(agent_id)
                        action = self.algorithm.compute_single_action(
                            obs[agent_id], 
                            policy_id=policy_id,
                            explore=False
                        )
                        actions[agent_id] = action
                
                # Step environment
                obs, rewards, done, infos = eval_env.step(actions)
                
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Collect custom metrics
                if infos:
                    market_efficiency = infos.get("market_efficiency", 0)
                    cooperation_score = infos.get("cooperation_score", 0)
                    violation_rate = infos.get("violation_rate", 0)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            market_efficiencies.append(market_efficiency)
            cooperation_scores.append(cooperation_score)
            violation_rates.append(violation_rate)
        
        eval_env.close()
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_market_efficiency": np.mean(market_efficiencies),
            "mean_cooperation_score": np.mean(cooperation_scores),
            "mean_violation_rate": np.mean(violation_rates)
        }
    
    def _check_curriculum_progression(self, iteration: int):
        """Check if we should progress to next curriculum stage"""
        
        if not self.config.curriculum_enabled:
            return
        
        current_stage = self.config.curriculum_stages[self.current_curriculum_stage]
        stage_iterations = current_stage.get('iterations', 100)
        
        if iteration >= sum(stage['iterations'] for stage in self.config.curriculum_stages[:self.current_curriculum_stage + 1]):
            if self.current_curriculum_stage < len(self.config.curriculum_stages) - 1:
                self.current_curriculum_stage += 1
                next_stage = self.config.curriculum_stages[self.current_curriculum_stage]
                print(f"Progressing to curriculum stage: {next_stage['name']}")
                
                # Update environment configuration
                self.algorithm.workers.foreach_worker(
                    lambda worker: worker.set_weights(self.algorithm.get_weights())
                )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint"""
        
        if self.algorithm is None:
            self.setup_algorithm()
        
        self.algorithm.restore(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        
        # Load training metrics if available
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.training_metrics = json.load(f)
            print("Loaded training metrics")
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        
        if self.algorithm is None:
            raise ValueError("No trained algorithm to save")
        
        checkpoint_path = self.algorithm.save(save_path)
        print(f"Model saved to: {checkpoint_path}")
        return checkpoint_path
    
    def hyperparameter_tuning(self, 
                             search_space: Dict[str, Any],
                             num_samples: int = 10,
                             max_concurrent: int = 4) -> Dict[str, Any]:
        """Perform hyperparameter tuning using Ray Tune"""
        
        def train_function(config_dict):
            # Update training config with tuned parameters
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Setup and train
            self.setup_algorithm()
            
            # Train for a subset of iterations
            tune_iterations = min(100, self.config.num_iterations)
            
            for i in range(tune_iterations):
                result = self.algorithm.train()
                
                # Report metrics to Tune
                tune.report(
                    episode_reward_mean=result.get("episode_reward_mean", 0),
                    market_efficiency=result.get("custom_metrics", {}).get("market_efficiency", 0)
                )
        
        # Run hyperparameter search
        analysis = tune.run(
            train_function,
            config=search_space,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            metric="episode_reward_mean",
            mode="max",
            verbose=1
        )
        
        best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
        print(f"Best hyperparameters: {best_config}")
        
        return best_config
    
    def close(self):
        """Clean up resources"""
        
        if self.algorithm:
            self.algorithm.stop()
        
        if self.config.wandb_project:
            wandb.finish()
        
        if ray.is_initialized():
            ray.shutdown()

def create_training_config(scenario: str = "balanced", **kwargs) -> TrainingConfig:
    """Create a training configuration for specific scenarios"""
    
    base_config = TrainingConfig(**kwargs)
    
    if scenario == "energy_trading":
        base_config.num_agents = 20
        base_config.max_episode_steps = 1500
        base_config.scenario = "energy_trading"
        
    elif scenario == "data_marketplace":
        base_config.num_agents = 15
        base_config.max_episode_steps = 1200
        base_config.scenario = "data_marketplace"
        
    elif scenario == "financial_market":
        base_config.num_agents = 30
        base_config.max_episode_steps = 2000
        base_config.scenario = "financial_market"
        base_config.learning_rate = 1e-4  # More conservative for financial markets
        
    elif scenario == "commodity_exchange":
        base_config.num_agents = 25
        base_config.max_episode_steps = 1800
        base_config.scenario = "commodity_exchange"
    
    return base_config

def main():
    """Main training function"""
    
    # Example usage
    config = create_training_config(
        scenario="balanced",
        algorithm="PPO",
        num_iterations=500,
        num_workers=4,
        wandb_project="ai-negotiator-test"
    )
    
    trainer = MARLTrainer(config)
    
    try:
        metrics = trainer.train()
        print("Training completed successfully!")
        
        # Save final model
        model_path = trainer.save_model("final_model")
        print(f"Final model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        trainer.close()

if __name__ == "__main__":
    main()