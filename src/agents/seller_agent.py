"""
Seller Agent Implementation
Focuses on maximizing profit through strategic pricing and inventory management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
import random

from .base_agent import BaseAgent, MarketplaceFeatureExtractor
from ..environment.marketplace import AgentType, ResourceType, MessageType, Message, Trade

class SellerPolicyNetwork(nn.Module):
    """Neural network for seller decision making"""
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = MarketplaceFeatureExtractor(observation_space, hidden_dim)
        
        # Seller-specific layers
        self.pricing_head = nn.Linear(hidden_dim, 1)  # Dynamic pricing
        self.inventory_management_head = nn.Linear(hidden_dim, len(ResourceType))  # Inventory decisions
        
        # Standard action heads
        self.trade_resource_head = nn.Linear(hidden_dim, len(ResourceType))
        self.trade_quantity_head = nn.Linear(hidden_dim, 1)
        self.trade_price_head = nn.Linear(hidden_dim, 1)
        self.trade_target_head = nn.Linear(hidden_dim, 10)
        self.trade_action_head = nn.Linear(hidden_dim, 4)
        
        self.comm_message_head = nn.Linear(hidden_dim, len(MessageType))
        self.comm_target_head = nn.Linear(hidden_dim, 10)
        self.comm_enabled_head = nn.Linear(hidden_dim, 2)
        
        self.alliance_action_head = nn.Linear(hidden_dim, 4)
        self.alliance_target_head = nn.Linear(hidden_dim, 10)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, observations):
        features = self.feature_extractor(observations)
        
        # Seller-specific outputs
        pricing_multiplier = torch.sigmoid(self.pricing_head(features)) * 2.0  # 0-2x market price
        inventory_weights = torch.softmax(self.inventory_management_head(features), dim=-1)
        
        # Standard outputs
        trade_resource_logits = self.trade_resource_head(features)
        trade_quantity = torch.sigmoid(self.trade_quantity_head(features)) * 1000
        trade_price = torch.sigmoid(self.trade_price_head(features)) * 10000
        trade_target_logits = self.trade_target_head(features)
        trade_action_logits = self.trade_action_head(features)
        
        comm_message_logits = self.comm_message_head(features)
        comm_target_logits = self.comm_target_head(features)
        comm_enabled_logits = self.comm_enabled_head(features)
        
        alliance_action_logits = self.alliance_action_head(features)
        alliance_target_logits = self.alliance_target_head(features)
        
        value = self.value_head(features)
        
        return {
            'pricing_multiplier': pricing_multiplier,
            'inventory_weights': inventory_weights,
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

class SellerAgent(BaseAgent):
    """
    Seller agent that maximizes profit through strategic pricing and inventory management
    Uses sophisticated pricing strategies and market analysis
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.SELLER, **kwargs)
        
        # Seller-specific parameters
        self.profit_margin_target = 0.3  # Target 30% profit margin
        self.inventory_turnover_target = 0.8  # Aim to sell 80% of inventory
        self.price_elasticity = 0.5  # How responsive to demand changes
        self.quality_premium = 0.2  # Premium for high quality resources
        self.bulk_discount = 0.1  # Discount for large orders
        
        # Pricing strategy
        self.pricing_strategy = 'dynamic'  # 'fixed', 'dynamic', 'competitive'
        self.base_markup = 0.25  # Base markup over cost
        self.min_markup = 0.1  # Minimum markup to maintain profitability
        self.max_markup = 0.8  # Maximum markup before losing customers
        
        # Market analysis
        self.demand_history = {rt: [] for rt in ResourceType}
        self.competitor_prices = {rt: [] for rt in ResourceType}
        self.sales_history = []
        self.customer_segments = {}  # Track different customer types
        
        # Initialize networks
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        
        # Learning parameters
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
    def initialize_networks(self, observation_space, action_space):
        """Initialize neural networks"""
        self.policy_network = SellerPolicyNetwork(observation_space, action_space)
        self.target_network = SellerPolicyNetwork(observation_space, action_space)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get action using neural network and seller-specific strategies"""
        
        if self.policy_network is None:
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
        
        action = {}
        
        # Trade actions with seller-specific logic
        if random.random() < self.epsilon:
            # Exploration
            action['trade_resource_type'] = random.randint(0, len(ResourceType) - 1)
            action['trade_quantity'] = [random.uniform(1, 100)]
            action['trade_price'] = [random.uniform(50, 2000)]
            action['trade_target'] = random.randint(0, 9)
            action['trade_action_type'] = random.choice([1, 2])  # Sell or no-action
        else:
            # Exploitation with seller strategy
            action = self._get_strategic_trade_action(outputs, observation)
        
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
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return action
    
    def _get_strategic_trade_action(self, outputs, observation):
        """Get strategic trade action based on seller objectives"""
        
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        cash = observation['cash'][0]
        
        action = {}
        
        # Choose resource with highest inventory and good profit potential
        inventory_weights = outputs['inventory_weights'].squeeze().cpu().numpy()
        resource_priorities = []
        
        for i, (inventory, market_price, weight) in enumerate(zip(portfolio, market_prices, inventory_weights)):
            if inventory > 5:  # Only sell if we have reasonable inventory
                # Calculate potential profit
                optimal_price = self._calculate_optimal_price(i, market_price, inventory)
                profit_potential = (optimal_price - market_price) * min(inventory, 50)
                
                priority = profit_potential * weight * inventory
                resource_priorities.append((i, priority, optimal_price))
        
        if resource_priorities:
            # Sort by priority and select best resource
            resource_priorities.sort(key=lambda x: x[1], reverse=True)
            best_resource_idx, _, optimal_price = resource_priorities[0]
            
            action['trade_resource_type'] = best_resource_idx
            action['trade_price'] = [optimal_price]
            
            # Determine quantity based on inventory and demand
            available_inventory = portfolio[best_resource_idx]
            target_quantity = min(available_inventory * 0.3, 100)  # Sell up to 30% of inventory
            action['trade_quantity'] = [max(1, target_quantity)]
            
            # Select target buyer (prefer buyers with good reputation)
            action['trade_target'] = self._select_target_buyer(observation)
            action['trade_action_type'] = 1  # Sell
        else:
            # No good selling opportunities
            action['trade_resource_type'] = 0
            action['trade_quantity'] = [0]
            action['trade_price'] = [0]
            action['trade_target'] = 0
            action['trade_action_type'] = 2  # No action
        
        return action
    
    def _get_heuristic_action(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback heuristic action"""
        
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        cash = observation['cash'][0]
        
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
        
        # Find resource with highest inventory
        max_quantity = 0
        best_resource_idx = 0
        for i, quantity in enumerate(portfolio):
            if quantity > max_quantity and quantity > 10:
                max_quantity = quantity
                best_resource_idx = i
        
        # Sell if we have significant inventory
        if max_quantity > 10:
            market_price = market_prices[best_resource_idx]
            selling_price = market_price * (1 + self.base_markup)  # Add markup
            
            action['trade_resource_type'] = best_resource_idx
            action['trade_quantity'] = [min(max_quantity * 0.2, 50)]  # Sell 20% of inventory
            action['trade_price'] = [selling_price]
            action['trade_target'] = random.randint(0, 9)  # Random buyer
            action['trade_action_type'] = 1  # Sell
        
        # Occasionally send promotional messages
        if random.random() < 0.15:
            action['comm_enabled'] = 1
            action['comm_message_type'] = 0  # Offer message
            action['comm_target'] = random.randint(0, 9)
        
        return action
    
    def _calculate_optimal_price(self, resource_idx: int, market_price: float, inventory: float) -> float:
        """Calculate optimal price using economic principles"""
        
        # Base price with markup
        base_price = market_price * (1 + self.base_markup)
        
        # Adjust for inventory levels (higher inventory = lower price to move stock)
        inventory_factor = max(0.8, 1.0 - (inventory - 50) / 200)  # Reduce price if high inventory
        
        # Adjust for demand (if we have demand history)
        demand_factor = 1.0
        resource_type = list(ResourceType)[resource_idx]
        if resource_type in self.demand_history and self.demand_history[resource_type]:
            recent_demand = np.mean(self.demand_history[resource_type][-5:])
            if recent_demand > 10:  # High demand
                demand_factor = 1.1
            elif recent_demand < 5:  # Low demand
                demand_factor = 0.9
        
        # Price elasticity adjustment
        elasticity_factor = 1.0 + (self.price_elasticity - 0.5) * 0.2
        
        # Calculate final price
        optimal_price = base_price * inventory_factor * demand_factor * elasticity_factor
        
        # Ensure price stays within reasonable bounds
        min_price = market_price * (1 + self.min_markup)
        max_price = market_price * (1 + self.max_markup)
        
        return np.clip(optimal_price, min_price, max_price)
    
    def _select_target_buyer(self, observation: Dict[str, np.ndarray]) -> int:
        """Select target buyer based on reputation and buying patterns"""
        
        other_agents_reputation = observation['other_agents_reputation']
        
        # Prefer buyers with higher reputation
        if len(other_agents_reputation) > 0:
            best_buyer_idx = np.argmax(other_agents_reputation)
            return best_buyer_idx
        
        return random.randint(0, 9)
    
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update seller policy using experiences"""
        
        if self.policy_network is None or len(experiences['observations']) < 32:
            return {'loss': 0.0}
        
        batch_size = min(32, len(experiences['observations']))
        indices = random.sample(range(len(experiences['observations'])), batch_size)
        
        # Prepare batch
        batch_obs = {}
        for key in experiences['observations'][0].keys():
            batch_obs[key] = torch.FloatTensor([experiences['observations'][i][key] for i in indices])
        
        batch_rewards = torch.FloatTensor([experiences['rewards'][i] for i in indices])
        batch_actions = [experiences['actions'][i] for i in indices]
        
        # Forward pass
        outputs = self.policy_network(batch_obs)
        
        # Calculate losses
        total_loss = 0.0
        
        # Value loss
        predicted_values = outputs['value'].squeeze()
        value_loss = nn.MSELoss()(predicted_values, batch_rewards)
        total_loss += value_loss
        
        # Policy losses with advantage weighting
        advantages = batch_rewards - predicted_values.detach()
        
        # Seller-specific losses
        for i, action in enumerate(batch_actions):
            if action['trade_action_type'] == 1:  # Selling action
                # Pricing loss (encourage profitable pricing)
                predicted_price = outputs['trade_price'][i]
                actual_price = torch.FloatTensor([action['trade_price'][0]])
                pricing_loss = nn.MSELoss()(predicted_price, actual_price)
                total_loss += pricing_loss * advantages[i] * 0.5
                
                # Resource selection loss
                resource_logits = outputs['trade_resource_logits'][i]
                resource_target = torch.LongTensor([action['trade_resource_type']])
                resource_loss = nn.CrossEntropyLoss()(resource_logits.unsqueeze(0), resource_target)
                total_loss += resource_loss * advantages[i]
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item()}
    
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate seller-specific message responses"""
        
        if message.message_type == MessageType.OFFER:
            # Buyer is making an offer to us
            content = message.content
            offered_price = content.get('price', 0)
            quantity = content.get('quantity', 0)
            resource_type = content.get('resource_type')
            
            # Evaluate the offer
            if resource_type and hasattr(ResourceType, resource_type.upper()):
                rt = ResourceType[resource_type.upper()]
                rt_idx = list(ResourceType).index(rt)
                
                # Check if we have the inventory
                if rt_idx < len(self.portfolio) and self.portfolio[rt_idx] >= quantity:
                    # Calculate our minimum acceptable price
                    min_price = self._calculate_optimal_price(rt_idx, offered_price, self.portfolio[rt_idx]) * 0.9
                    
                    if offered_price >= min_price:
                        return self.generate_message(
                            message.sender,
                            MessageType.ACCEPT,
                            {'trade_id': content.get('trade_id'), 'accepted': True}
                        )
                    else:
                        # Counter with our preferred price
                        counter_price = min_price * 1.1
                        return self.generate_message(
                            message.sender,
                            MessageType.COUNTER_OFFER,
                            {
                                'trade_id': content.get('trade_id'),
                                'counter_price': counter_price,
                                'quantity': quantity
                            }
                        )
                
                return self.generate_message(
                    message.sender,
                    MessageType.REJECT,
                    {'trade_id': content.get('trade_id'), 'reason': 'insufficient_inventory'}
                )
        
        elif message.message_type == MessageType.INFORMATION_REQUEST:
            # Share market information to build relationships
            if self.should_trust_agent(message.sender, 0.4):  # Lower trust threshold for sellers
                info_type = message.content.get('info_type', 'inventory')
                
                if info_type == 'inventory':
                    # Share partial inventory information
                    inventory_info = {}
                    for rt in ResourceType:
                        rt_idx = list(ResourceType).index(rt)
                        if rt_idx < len(self.portfolio) and self.portfolio[rt_idx] > 10:
                            inventory_info[rt.value] = 'available'
                        else:
                            inventory_info[rt.value] = 'limited'
                    
                    return self.generate_message(
                        message.sender,
                        MessageType.INFORMATION_SHARE,
                        {'inventory_status': inventory_info}
                    )
        
        elif message.message_type == MessageType.ALLIANCE_PROPOSAL:
            # Consider alliance if it could increase market share
            if self.cooperation_tendency > 0.6 and self.should_trust_agent(message.sender, 0.6):
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
        
        return None
    
    def analyze_market_trends(self, observation: Dict[str, np.ndarray]):
        """Analyze market trends to adjust strategy"""
        
        market_prices = observation['market_prices']
        time_step = observation['time_step'][0]
        
        # Update price history
        for i, price in enumerate(market_prices):
            resource_type = list(ResourceType)[i]
            self.competitor_prices[resource_type].append(price)
            
            # Keep only recent history
            if len(self.competitor_prices[resource_type]) > 50:
                self.competitor_prices[resource_type] = self.competitor_prices[resource_type][-50:]
        
        # Adjust pricing strategy based on trends
        for resource_type in ResourceType:
            if resource_type in self.competitor_prices and len(self.competitor_prices[resource_type]) > 5:
                prices = self.competitor_prices[resource_type]
                
                # Calculate trend
                recent_avg = np.mean(prices[-5:])
                older_avg = np.mean(prices[-10:-5]) if len(prices) >= 10 else recent_avg
                
                if recent_avg > older_avg * 1.05:  # Prices rising
                    self.base_markup = min(self.max_markup, self.base_markup * 1.02)
                elif recent_avg < older_avg * 0.95:  # Prices falling
                    self.base_markup = max(self.min_markup, self.base_markup * 0.98)
    
    def update_customer_segments(self, buyer_id: str, trade: Trade):
        """Update customer segmentation based on trading patterns"""
        
        if buyer_id not in self.customer_segments:
            self.customer_segments[buyer_id] = {
                'total_trades': 0,
                'total_volume': 0,
                'avg_price_paid': 0,
                'preferred_resources': {},
                'loyalty_score': 0.5
            }
        
        segment = self.customer_segments[buyer_id]
        segment['total_trades'] += 1
        segment['total_volume'] += trade.resource.quantity
        
        # Update average price
        old_avg = segment['avg_price_paid']
        segment['avg_price_paid'] = (old_avg * (segment['total_trades'] - 1) + trade.agreed_price) / segment['total_trades']
        
        # Update preferred resources
        resource_type = trade.resource.type.value
        if resource_type not in segment['preferred_resources']:
            segment['preferred_resources'][resource_type] = 0
        segment['preferred_resources'][resource_type] += 1
        
        # Update loyalty score based on repeat business
        if segment['total_trades'] > 5:
            segment['loyalty_score'] = min(1.0, segment['loyalty_score'] + 0.05)
    
    def get_inventory_turnover_rate(self) -> float:
        """Calculate inventory turnover rate"""
        if not self.sales_history:
            return 0.0
        
        recent_sales = sum(trade.resource.quantity for trade in self.sales_history[-10:])
        current_inventory = sum(self.portfolio.values())
        
        if current_inventory == 0:
            return 1.0  # All inventory sold
        
        return min(1.0, recent_sales / current_inventory)