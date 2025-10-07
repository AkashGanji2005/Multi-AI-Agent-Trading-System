"""
Base agent class for the marketplace environment
Provides common functionality for all agent types
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from ..environment.marketplace import AgentType, ResourceType, MessageType, Message, Trade, Alliance

@dataclass
class AgentMemory:
    """Memory structure for agent experiences"""
    observations: List[Dict[str, np.ndarray]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    next_observations: List[Dict[str, np.ndarray]]
    dones: List[bool]
    messages: List[Message]
    trades: List[Trade]
    
    def __post_init__(self):
        self.max_size = 10000
    
    def add_experience(self, obs, action, reward, next_obs, done, messages=None, trades=None):
        """Add experience to memory"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.messages.append(messages or [])
        self.trades.append(trades or [])
        
        # Keep memory size bounded
        if len(self.observations) > self.max_size:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_observations.pop(0)
            self.dones.pop(0)
            self.messages.pop(0)
            self.trades.pop(0)
    
    def get_recent_experiences(self, n: int = 100):
        """Get recent experiences"""
        return {
            'observations': self.observations[-n:],
            'actions': self.actions[-n:],
            'rewards': self.rewards[-n:],
            'next_observations': self.next_observations[-n:],
            'dones': self.dones[-n:],
            'messages': self.messages[-n:],
            'trades': self.trades[-n:]
        }

class MarketplaceFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for marketplace observations
    Handles the complex Dict observation space
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        self.cash_dim = 1
        self.portfolio_dim = observation_space['portfolio'].shape[0]
        self.reputation_dim = 1
        self.agent_type_dim = 1
        self.market_prices_dim = observation_space['market_prices'].shape[0]
        self.time_step_dim = 1
        self.other_agents_rep_dim = observation_space['other_agents_reputation'].shape[0]
        self.other_agents_types_dim = observation_space['other_agents_types'].shape[0]
        self.recent_messages_dim = np.prod(observation_space['recent_messages'].shape)
        self.active_trades_dim = np.prod(observation_space['active_trades'].shape)
        self.alliance_memberships_dim = observation_space['alliance_memberships'].shape[0]
        
        total_input_dim = (self.cash_dim + self.portfolio_dim + self.reputation_dim + 
                          self.agent_type_dim + self.market_prices_dim + self.time_step_dim +
                          self.other_agents_rep_dim + self.other_agents_types_dim +
                          self.recent_messages_dim + self.active_trades_dim + 
                          self.alliance_memberships_dim)
        
        # Neural network layers
        self.linear_layers = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations"""
        
        # Flatten and concatenate all observation components
        features = []
        
        features.append(observations['cash'].float())
        features.append(observations['portfolio'].float())
        features.append(observations['reputation'].float())
        features.append(observations['agent_type'].float())
        features.append(observations['market_prices'].float())
        features.append(observations['time_step'].float())
        features.append(observations['other_agents_reputation'].float())
        features.append(observations['other_agents_types'].float())
        features.append(observations['recent_messages'].flatten().float())
        features.append(observations['active_trades'].flatten().float())
        features.append(observations['alliance_memberships'].float())
        
        # Concatenate all features
        concatenated = torch.cat(features, dim=-1)
        
        # Pass through neural network
        return self.linear_layers(concatenated)

class BaseAgent(ABC):
    """
    Abstract base class for all marketplace agents
    Provides common functionality and interface
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: AgentType,
                 initial_cash: float = 1000.0,
                 initial_portfolio: Dict[ResourceType, float] = None,
                 objectives: Dict[str, float] = None,
                 learning_rate: float = 3e-4,
                 memory_size: int = 10000):
        
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initial_cash = initial_cash
        self.initial_portfolio = initial_portfolio or {}
        self.objectives = objectives or {}
        self.learning_rate = learning_rate
        
        # Agent state
        self.cash = initial_cash
        self.portfolio = initial_portfolio.copy() if initial_portfolio else {}
        self.reputation = 0.5
        self.violations = 0
        
        # Memory and learning
        self.memory = AgentMemory([], [], [], [], [], [], [])
        self.step_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Communication and social
        self.trusted_agents = set()
        self.blacklisted_agents = set()
        self.alliance_memberships = []
        self.pending_messages = []
        self.message_history = []
        
        # Strategy parameters (can be learned)
        self.risk_tolerance = 0.5
        self.negotiation_patience = 0.7
        self.cooperation_tendency = 0.6
        self.price_aggressiveness = 0.5
        
        # Performance tracking
        self.trade_success_rate = 0.0
        self.total_trades_attempted = 0
        self.total_trades_completed = 0
        self.average_profit_per_trade = 0.0
        self.total_profit = 0.0
    
    @abstractmethod
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get action based on observation"""
        pass
    
    @abstractmethod
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent's policy based on experiences"""
        pass
    
    def reset(self):
        """Reset agent state for new episode"""
        self.cash = self.initial_cash
        self.portfolio = self.initial_portfolio.copy() if self.initial_portfolio else {}
        self.reputation = 0.5
        self.violations = 0
        self.current_episode_reward = 0.0
        self.pending_messages = []
        
        # Reset some strategy parameters with noise for exploration
        noise_scale = 0.1
        self.risk_tolerance = np.clip(self.risk_tolerance + np.random.normal(0, noise_scale), 0, 1)
        self.negotiation_patience = np.clip(self.negotiation_patience + np.random.normal(0, noise_scale), 0, 1)
        self.cooperation_tendency = np.clip(self.cooperation_tendency + np.random.normal(0, noise_scale), 0, 1)
        self.price_aggressiveness = np.clip(self.price_aggressiveness + np.random.normal(0, noise_scale), 0, 1)
    
    def add_experience(self, obs, action, reward, next_obs, done, info=None):
        """Add experience to memory"""
        self.memory.add_experience(obs, action, reward, next_obs, done,
                                 info.get('messages', []) if info else [],
                                 info.get('trades', []) if info else [])
        self.current_episode_reward += reward
        self.step_count += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
    
    def calculate_portfolio_value(self, market_prices: Dict[ResourceType, float]) -> float:
        """Calculate total portfolio value at current market prices"""
        total_value = self.cash
        for resource_type, quantity in self.portfolio.items():
            if resource_type in market_prices:
                total_value += quantity * market_prices[resource_type]
        return total_value
    
    def assess_trade_opportunity(self, trade: Trade, market_prices: Dict[ResourceType, float]) -> float:
        """Assess the attractiveness of a trade opportunity"""
        resource_type = trade.resource.type
        market_price = market_prices.get(resource_type, trade.agreed_price)
        
        if trade.buyer == self.agent_id:
            # Buying - good if price is below market
            value = (market_price - trade.agreed_price) / market_price
        else:
            # Selling - good if price is above market  
            value = (trade.agreed_price - market_price) / market_price
        
        # Adjust for resource quality
        quality_bonus = (trade.resource.quality - 0.5) * 0.2
        
        # Adjust for urgency (time-sensitive trades)
        urgency_factor = 1.0
        if hasattr(trade.resource, 'expiry_time') and trade.resource.expiry_time:
            urgency_factor = max(0.5, trade.resource.expiry_time / 100)
        
        return value + quality_bonus * urgency_factor
    
    def should_trust_agent(self, other_agent_id: str, reputation: float) -> bool:
        """Determine if another agent should be trusted"""
        if other_agent_id in self.blacklisted_agents:
            return False
        
        if other_agent_id in self.trusted_agents:
            return True
        
        # Base trust on reputation and cooperation tendency
        trust_threshold = 1.0 - self.cooperation_tendency
        return reputation > trust_threshold
    
    def update_agent_relationship(self, other_agent_id: str, positive: bool):
        """Update relationship with another agent based on interaction outcome"""
        if positive:
            self.trusted_agents.add(other_agent_id)
            self.blacklisted_agents.discard(other_agent_id)
        else:
            self.blacklisted_agents.add(other_agent_id)
            self.trusted_agents.discard(other_agent_id)
    
    def generate_message(self, receiver: str, message_type: MessageType, content: Dict[str, Any]) -> Message:
        """Generate a message to send to another agent"""
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=self.step_count
        )
        self.message_history.append(message)
        return message
    
    def process_received_message(self, message: Message) -> Optional[Message]:
        """Process a received message and optionally generate a response"""
        self.pending_messages.append(message)
        
        # Update relationship based on message type
        if message.message_type in [MessageType.ACCEPT, MessageType.ALLIANCE_ACCEPT]:
            self.update_agent_relationship(message.sender, True)
        elif message.message_type in [MessageType.REJECT, MessageType.ALLIANCE_REJECT]:
            self.update_agent_relationship(message.sender, False)
        
        # Generate response based on agent type and message content
        return self._generate_message_response(message)
    
    @abstractmethod
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate response to a message (agent-specific implementation)"""
        pass
    
    def calculate_expected_utility(self, action: Dict[str, Any], observation: Dict[str, np.ndarray]) -> float:
        """Calculate expected utility of an action (game theory component)"""
        
        # Extract relevant information from observation
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        reputation = observation['reputation'][0]
        
        utility = 0.0
        
        # Trade action utility
        if action.get('trade_action_type', 2) != 2:  # Not no-action
            resource_idx = action.get('trade_resource_type', 0)
            quantity = action.get('trade_quantity', [0])[0]
            price = action.get('trade_price', [0])[0]
            
            if resource_idx < len(market_prices):
                market_price = market_prices[resource_idx]
                
                if action['trade_action_type'] == 0:  # Buy
                    # Utility of buying below market price
                    if price < market_price:
                        utility += (market_price - price) * quantity / 1000
                    # Penalty for insufficient cash
                    if cash < price * quantity:
                        utility -= 1.0
                        
                elif action['trade_action_type'] == 1:  # Sell
                    # Utility of selling above market price
                    if price > market_price:
                        utility += (price - market_price) * quantity / 1000
                    # Penalty for insufficient inventory
                    if resource_idx < len(portfolio) and portfolio[resource_idx] < quantity:
                        utility -= 1.0
        
        # Communication utility (small positive for maintaining relationships)
        if action.get('comm_enabled', 0):
            utility += 0.1 * self.cooperation_tendency
        
        # Alliance utility (depends on agent type and cooperation tendency)
        alliance_action = action.get('alliance_action', 0)
        if alliance_action == 1:  # Propose alliance
            utility += 0.2 * self.cooperation_tendency
        elif alliance_action == 2:  # Accept alliance
            utility += 0.15 * self.cooperation_tendency
        
        # Risk adjustment
        utility *= (1.0 - self.risk_tolerance * 0.5)
        
        return utility
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics"""
        return {
            'total_reward': sum(self.episode_rewards),
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'trade_success_rate': self.trade_success_rate,
            'total_profit': self.total_profit,
            'average_profit_per_trade': self.average_profit_per_trade,
            'reputation': self.reputation,
            'violations': self.violations,
            'cash': self.cash,
            'portfolio_size': sum(self.portfolio.values()),
            'trusted_agents': len(self.trusted_agents),
            'blacklisted_agents': len(self.blacklisted_agents)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save agent state for checkpointing"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'cash': self.cash,
            'portfolio': dict(self.portfolio),
            'reputation': self.reputation,
            'violations': self.violations,
            'risk_tolerance': self.risk_tolerance,
            'negotiation_patience': self.negotiation_patience,
            'cooperation_tendency': self.cooperation_tendency,
            'price_aggressiveness': self.price_aggressiveness,
            'episode_rewards': self.episode_rewards,
            'trade_success_rate': self.trade_success_rate,
            'total_profit': self.total_profit,
            'trusted_agents': list(self.trusted_agents),
            'blacklisted_agents': list(self.blacklisted_agents)
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.cash = state['cash']
        self.portfolio = state['portfolio']
        self.reputation = state['reputation']
        self.violations = state['violations']
        self.risk_tolerance = state['risk_tolerance']
        self.negotiation_patience = state['negotiation_patience']
        self.cooperation_tendency = state['cooperation_tendency']
        self.price_aggressiveness = state['price_aggressiveness']
        self.episode_rewards = state['episode_rewards']
        self.trade_success_rate = state['trade_success_rate']
        self.total_profit = state['total_profit']
        self.trusted_agents = set(state['trusted_agents'])
        self.blacklisted_agents = set(state['blacklisted_agents'])