#!/usr/bin/env python3
"""
Standalone training script for AI Negotiator
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.marl_trainer import MARLTrainer, create_training_config

def main():
    parser = argparse.ArgumentParser(description="Train AI Negotiator agents")
    parser.add_argument('--scenario', default='balanced', help='Training scenario')
    parser.add_argument('--algorithm', default='PPO', choices=['PPO', 'DQN', 'SAC'], help='RL algorithm')
    parser.add_argument('--iterations', type=int, default=1000, help='Training iterations')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--wandb-project', help='Weights & Biases project name')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Create training configuration
    config = create_training_config(
        scenario=args.scenario,
        algorithm=args.algorithm,
        num_iterations=args.iterations,
        num_workers=args.workers,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Create and run trainer
    trainer = MARLTrainer(config)
    
    try:
        print(f"Starting training with {args.algorithm} on {args.scenario} scenario...")
        metrics = trainer.train()
        
        # Save final model
        model_path = trainer.save_model(f"models/{args.scenario}_{args.algorithm}_final")
        print(f"Training completed! Model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        trainer.close()

if __name__ == "__main__":
    main()