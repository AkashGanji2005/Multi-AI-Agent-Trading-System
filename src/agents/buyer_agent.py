"""
Buyer Agent Implementation
Focuses on acquiring resources at minimum cost while maintaining quality standards
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
import random

from .base_agent import BaseAgent, MarketplaceFeatureExtractor
from ..environment.marketplace import AgentType, ResourceType, MessageType, Message, Trade

class BuyerPolicyNetwork(nn.Module):
    """Neural network for buyer decision making"""
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = MarketplaceFeatureExtractor(observation_space, hidden_dim)
        
        # Action heads
        self.trade_resource_head = nn.Linear(hidden_dim, len(ResourceType))
        self.trade_quantity_head = nn.Linear(hidden_dim, 1)
        self.trade_price_head = nn.Linear(hidden_dim, 1)
        self.trade_target_head = nn.Linear(hidden_dim, 10)  # Max 10 agents for now
        self.trade_action_head = nn.Linear(hidden_dim, 4)
        
        self.comm_message_head = nn.Linear(hidden_dim, len(MessageType))
        self.comm_target_head = nn.Linear(hidden_dim, 10)
        self.comm_enabled_head = nn.Linear(hidden_dim, 2)
        
        self.alliance_action_head = nn.Linear(hidden_dim, 4)
        self.alliance_target_head = nn.Linear(hidden_dim, 10)
        
        # Value function for advantage estimation
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, observations):
        features = self.feature_extractor(observations)
        
        # Trade actions
        trade_resource_logits = self.trade_resource_head(features)
        trade_quantity = torch.sigmoid(self.trade_quantity_head(features)) * 1000  # Max quantity 1000
        trade_price = torch.sigmoid(self.trade_price_head(features)) * 10000  # Max price 10000
        trade_target_logits = self.trade_target_head(features)
        trade_action_logits = self.trade_action_head(features)
        
        # Communication actions
        comm_message_logits = self.comm_message_head(features)
        comm_target_logits = self.comm_target_head(features)
        comm_enabled_logits = self.comm_enabled_head(features)
        
        # Alliance actions
        alliance_action_logits = self.alliance_action_head(features)
        alliance_target_logits = self.alliance_target_head(features)
        
        # Value estimation
        value = self.value_head(features)
        
        return {
            'trade_resource_logits': trade_resource_logits,
            'trade_quantity': trade_quantity,
            'trade_price': trade_price,
            'trade_target_logits': trade_target_logits,
            'trade_action_logits': trade_action_logits,
            'comm_message_logits': comm_message_logits,
            'comm_target_logits': comm_target_logits,
            'comm_enabled_logits': comm_enabled_logits,
            'alliance_action_logits': alliance_action_logits,
            'alliance_target_logits': alliance_target_logits,
            'value': value
        }

class BuyerAgent(BaseAgent):
    """
    Buyer agent that seeks to minimize costs while securing necessary resources
    Uses deep reinforcement learning with game-theoretic considerations
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.BUYER, **kwargs)
        
        # Buyer-specific parameters
        self.quality_threshold = 0.6  # Minimum acceptable quality
        self.urgency_factor = 0.8  # How urgently resources are needed
        self.budget_constraint = self.initial_cash * 0.8  # Don't spend all cash
        self.preferred_resources = kwargs.get('preferred_resources', list(ResourceType))
        self.max_price_premium = 0.2  # Maximum premium over market price
        
        # Learning parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Initialize networks (will be set when observation/action spaces are known)
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        
        # Buyer-specific memory
        self.successful_trades = []
        self.failed_negotiations = []
        self.preferred_sellers = set()
        self.price_history = {rt: [] for rt in ResourceType}
        
    def initialize_networks(self, observation_space, action_space):
        """Initialize neural networks once spaces are known"""
        self.policy_network = BuyerPolicyNetwork(observation_space, action_space)
        self.target_network = BuyerPolicyNetwork(observation_space, action_space)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get action based on observation using neural network and heuristics"""
        
        if self.policy_network is None:
            # Fallback to heuristic action if networks not initialized
            return self._get_heuristic_action(observation)
        
        # Convert observation to tensors
        obs_tensors = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.FloatTensor(value).unsqueeze(0)
            else:
                obs_tensors[key] = torch.FloatTensor([value]).unsqueeze(0)
        
        # Get network outputs
        with torch.no_grad():
            outputs = self.policy_network(obs_tensors)
        
        # Sample actions
        action = {}
        
        # Trade actions
        if random.random() < self.epsilon:
            # Exploration
            action['trade_resource_type'] = random.randint(0, len(ResourceType) - 1)
            action['trade_quantity'] = [random.uniform(1, 100)]
            action['trade_price'] = [random.uniform(10, 1000)]
            action['trade_target'] = random.randint(0, 9)
            action['trade_action_type'] = random.randint(0, 3)
        else:
            # Exploitation
            trade_resource_probs = torch.softmax(outputs['trade_resource_logits'], dim=-1)
            action['trade_resource_type'] = torch.multinomial(trade_resource_probs, 1).item()
            action['trade_quantity'] = outputs['trade_quantity'].cpu().numpy()
            action['trade_price'] = outputs['trade_price'].cpu().numpy()
            
            trade_target_probs = torch.softmax(outputs['trade_target_logits'], dim=-1)
            action['trade_target'] = torch.multinomial(trade_target_probs, 1).item()
            
            trade_action_probs = torch.softmax(outputs['trade_action_logits'], dim=-1)
            action['trade_action_type'] = torch.multinomial(trade_action_probs, 1).item()
        
        # Apply buyer-specific logic
        action = self._apply_buyer_heuristics(action, observation)
        
        # Communication actions
        comm_enabled_probs = torch.softmax(outputs['comm_enabled_logits'], dim=-1)
        action['comm_enabled'] = torch.multinomial(comm_enabled_probs, 1).item()
        
        if action['comm_enabled']:
            comm_message_probs = torch.softmax(outputs['comm_message_logits'], dim=-1)
            action['comm_message_type'] = torch.multinomial(comm_message_probs, 1).item()
            
            comm_target_probs = torch.softmax(outputs['comm_target_logits'], dim=-1)
            action['comm_target'] = torch.multinomial(comm_target_probs, 1).item()
        else:
            action['comm_message_type'] = 0
            action['comm_target'] = 0
        
        # Alliance actions
        alliance_action_probs = torch.softmax(outputs['alliance_action_logits'], dim=-1)
        action['alliance_action'] = torch.multinomial(alliance_action_probs, 1).item()
        
        alliance_target_probs = torch.softmax(outputs['alliance_target_logits'], dim=-1)
        action['alliance_target'] = torch.multinomial(alliance_target_probs, 1).item()
        
        # Decay exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return action
    
    def _get_heuristic_action(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback heuristic action when neural network is not available"""
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        reputation = observation['reputation'][0]
        
        action = {
            'trade_resource_type': 0,
            'trade_quantity': [0],
            'trade_price': [0],
            'trade_target': 0,
            'trade_action_type': 2,  # No action by default
            'comm_enabled': 0,
            'comm_message_type': 0,
            'comm_target': 0,
            'alliance_action': 0,
            'alliance_target': 0
        }
        
        # Find resource with lowest inventory
        min_quantity = float('inf')
        target_resource_idx = 0
        for i, quantity in enumerate(portfolio):
            if quantity < min_quantity:
                min_quantity = quantity
                target_resource_idx = i
        
        # Only buy if we have cash and need the resource
        if cash > 100 and min_quantity < 50:
            market_price = market_prices[target_resource_idx]
            max_affordable_quantity = min(100, cash / (market_price * 1.1))  # 10% buffer
            
            if max_affordable_quantity >= 1:
                action['trade_resource_type'] = target_resource_idx
                action['trade_quantity'] = [max_affordable_quantity * 0.5]  # Conservative quantity
                action['trade_price'] = [market_price * (1 - self.max_price_premium)]  # Try to buy below market
                action['trade_target'] = random.randint(0, 9)  # Random seller
                action['trade_action_type'] = 0  # Buy
        
        # Occasionally communicate to gather information
        if random.random() < 0.1:
            action['comm_enabled'] = 1
            action['comm_message_type'] = 7  # Information request
            action['comm_target'] = random.randint(0, 9)
        
        return action
    
    def _apply_buyer_heuristics(self, action: Dict[str, Any], observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Apply buyer-specific heuristics to modify actions"""
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        
        # Only execute buy actions
        if action['trade_action_type'] not in [0, 2]:  # Only buy or no-action
            action['trade_action_type'] = 2
        
        # Ensure we don't overspend
        if action['trade_action_type'] == 0:  # Buy
            resource_idx = action['trade_resource_type']
            quantity = action['trade_quantity'][0]
            price = action['trade_price'][0]
            total_cost = quantity * price
            
            # Check budget constraint
            if total_cost > cash * 0.8:  # Don't spend more than 80% of cash
                if cash > 100:
                    max_quantity = (cash * 0.8) / price
                    action['trade_quantity'] = [max(1, max_quantity)]
                else:
                    action['trade_action_type'] = 2  # No action if insufficient funds
            
            # Don't pay too much above market price
            if resource_idx < len(market_prices):
                market_price = market_prices[resource_idx]
                max_acceptable_price = market_price * (1 + self.max_price_premium)
                if price > max_acceptable_price:
                    action['trade_price'] = [max_acceptable_price]
            
            # Prioritize resources we have less of
            current_quantity = portfolio[resource_idx] if resource_idx < len(portfolio) else 0
            if current_quantity > 100:  # We have enough of this resource
                # Find a resource we need more
                min_quantity = float('inf')
                needed_resource_idx = resource_idx
                for i, qty in enumerate(portfolio):
                    if qty < min_quantity:
                        min_quantity = qty
                        needed_resource_idx = i
                action['trade_resource_type'] = needed_resource_idx
        
        return action
    
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update the buyer's policy using experiences"""
        
        if self.policy_network is None or len(experiences['observations']) < 32:
            return {'loss': 0.0}
        
        # Sample batch
        batch_size = min(32, len(experiences['observations']))
        indices = random.sample(range(len(experiences['observations'])), batch_size)
        
        # Prepare batch tensors
        batch_obs = {}
        for key in experiences['observations'][0].keys():
            batch_obs[key] = torch.FloatTensor([experiences['observations'][i][key] for i in indices])
        
        batch_rewards = torch.FloatTensor([experiences['rewards'][i] for i in indices])
        batch_actions = [experiences['actions'][i] for i in indices]
        
        # Forward pass
        outputs = self.policy_network(batch_obs)
        
        # Calculate losses
        total_loss = 0.0
        
        # Value loss (MSE with discounted rewards)
        predicted_values = outputs['value'].squeeze()
        value_loss = nn.MSELoss()(predicted_values, batch_rewards)
        total_loss += value_loss
        
        # Policy losses (using REINFORCE-like approach)
        advantages = batch_rewards - predicted_values.detach()
        
        # Trade action losses
        for i, action in enumerate(batch_actions):
            # Resource type loss
            resource_logits = outputs['trade_resource_logits'][i]
            resource_target = torch.LongTensor([action['trade_resource_type']])
            resource_loss = nn.CrossEntropyLoss()(resource_logits.unsqueeze(0), resource_target)
            total_loss += resource_loss * advantages[i]
            
            # Action type loss
            action_logits = outputs['trade_action_logits'][i]
            action_target = torch.LongTensor([action['trade_action_type']])
            action_loss = nn.CrossEntropyLoss()(action_logits.unsqueeze(0), action_target)
            total_loss += action_loss * advantages[i]
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network occasionally
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item()}
    
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate response to received message"""
        
        if message.message_type == MessageType.OFFER:
            # Evaluate the offer
            content = message.content
            resource_type = content.get('resource_type')
            quantity = content.get('quantity', 0)
            price = content.get('price', 0)
            
            # Simple acceptance logic based on current needs and budget
            if self.cash >= price * quantity:
                if resource_type in self.preferred_resources:
                    return self.generate_message(
                        message.sender,
                        MessageType.ACCEPT,
                        {'trade_id': content.get('trade_id'), 'accepted': True}
                    )
                else:
                    # Counter-offer with lower price
                    counter_price = price * 0.9
                    return self.generate_message(
                        message.sender,
                        MessageType.COUNTER_OFFER,
                        {
                            'trade_id': content.get('trade_id'),
                            'counter_price': counter_price,
                            'quantity': quantity
                        }
                    )
            else:
                return self.generate_message(
                    message.sender,
                    MessageType.REJECT,
                    {'trade_id': content.get('trade_id'), 'reason': 'insufficient_funds'}
                )
        
        elif message.message_type == MessageType.ALLIANCE_PROPOSAL:
            # Consider alliance based on sender's reputation and our cooperation tendency
            if self.should_trust_agent(message.sender, 0.6):  # Assume moderate reputation
                if random.random() < self.cooperation_tendency:
                    return self.generate_message(
                        message.sender,
                        MessageType.ALLIANCE_ACCEPT,
                        {'alliance_id': message.content.get('alliance_id')}
                    )
            
            return self.generate_message(
                message.sender,
                MessageType.ALLIANCE_REJECT,
                {'alliance_id': message.content.get('alliance_id')}
            )
        
        elif message.message_type == MessageType.INFORMATION_REQUEST:
            # Share information if we trust the agent
            if self.should_trust_agent(message.sender, 0.5):
                info_type = message.content.get('info_type', 'market_prices')
                if info_type == 'market_prices':
                    return self.generate_message(
                        message.sender,
                        MessageType.INFORMATION_SHARE,
                        {'price_history': {rt.value: prices[-5:] for rt, prices in self.price_history.items()}}
                    )
        
        return None
    
    def evaluate_seller_performance(self, seller_id: str, trade: Trade) -> float:
        """Evaluate a seller's performance after a completed trade"""
        
        # Factors: price competitiveness, quality, delivery time, reliability
        score = 0.0
        
        # Price competitiveness (compared to market)
        market_price = self.price_history.get(trade.resource.type, [trade.agreed_price])[-1]
        if market_price > 0:
            price_score = max(0, 1 - (trade.agreed_price - market_price) / market_price)
            score += price_score * 0.4
        
        # Quality score
        quality_score = trade.resource.quality
        score += quality_score * 0.3
        
        # Reliability (did trade complete successfully?)
        if trade.status == "completed":
            score += 0.3
        
        # Update preferred sellers
        if score > 0.7:
            self.preferred_sellers.add(seller_id)
        elif score < 0.3:
            self.preferred_sellers.discard(seller_id)
        
        return score
    
    def get_resource_priority(self, observation: Dict[str, np.ndarray]) -> List[int]:
        """Get prioritized list of resources to acquire"""
        
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        
        # Calculate priority based on scarcity and strategic value
        priorities = []
        for i, (quantity, price) in enumerate(zip(portfolio, market_prices)):
            # Higher priority for scarce resources
            scarcity_score = max(0, (50 - quantity) / 50)
            
            # Lower priority for expensive resources (budget consideration)
            affordability_score = max(0, 1 - (price / 1000))
            
            # Strategic value based on agent objectives
            strategic_value = self.objectives.get('minimize_cost', 0.8)
            
            priority = scarcity_score * 0.5 + affordability_score * 0.3 + strategic_value * 0.2
            priorities.append((i, priority))
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in priorities]