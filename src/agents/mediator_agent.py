"""
Mediator Agent Implementation
Facilitates negotiations, resolves disputes, and earns fees for mediation services
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import defaultdict

from .base_agent import BaseAgent, MarketplaceFeatureExtractor
from ..environment.marketplace import AgentType, ResourceType, MessageType, Message, Trade

class MediatorPolicyNetwork(nn.Module):
    """Neural network for mediator decision making"""
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = MarketplaceFeatureExtractor(observation_space, hidden_dim)
        
        # Mediator-specific heads
        self.mediation_strategy_head = nn.Linear(hidden_dim, 4)  # Compromise, favor_buyer, favor_seller, neutral
        self.fee_structure_head = nn.Linear(hidden_dim, 3)      # Low, medium, high fee
        self.conflict_priority_head = nn.Linear(hidden_dim, 5)  # Priority levels for conflicts
        self.negotiation_style_head = nn.Linear(hidden_dim, 3)  # Aggressive, balanced, diplomatic
        
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
        
        # Mediator-specific outputs
        mediation_strategy_logits = self.mediation_strategy_head(features)
        fee_structure_logits = self.fee_structure_head(features)
        conflict_priority_logits = self.conflict_priority_head(features)
        negotiation_style_logits = self.negotiation_style_head(features)
        
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
            'mediation_strategy_logits': mediation_strategy_logits,
            'fee_structure_logits': fee_structure_logits,
            'conflict_priority_logits': conflict_priority_logits,
            'negotiation_style_logits': negotiation_style_logits,
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

class MediatorAgent(BaseAgent):
    """
    Mediator agent that facilitates negotiations and resolves disputes
    Earns fees for successful mediation and builds reputation through fair resolutions
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.MEDIATOR, **kwargs)
        
        # Mediator-specific parameters
        self.base_mediation_fee = 50.0  # Base fee for mediation services
        self.success_bonus_rate = 0.1   # Bonus as percentage of trade value
        self.reputation_weight = 0.8    # How much reputation affects fee pricing
        self.dispute_resolution_time = 5  # Average steps to resolve disputes
        
        # Mediation strategies
        self.mediation_styles = {
            'compromise': 0.5,      # Split the difference
            'favor_buyer': 0.3,     # Slightly favor buyers
            'favor_seller': 0.7,    # Slightly favor sellers
            'neutral': 0.5          # Completely neutral
        }
        
        # Fee structure
        self.fee_levels = {
            'low': 0.8,     # 80% of base fee
            'medium': 1.0,  # 100% of base fee
            'high': 1.5     # 150% of base fee
        }
        
        # Active mediations
        self.active_mediations = {}
        self.mediation_history = []
        self.dispute_queue = []
        self.success_rate = 0.7  # Initial success rate
        
        # Client relationships
        self.client_history = defaultdict(list)
        self.client_satisfaction = defaultdict(float)
        self.repeat_clients = set()
        
        # Negotiation tactics
        self.negotiation_tactics = {
            'information_gathering': 0.8,
            'creative_solutions': 0.7,
            'deadline_pressure': 0.6,
            'relationship_building': 0.9
        }
        
        # Networks
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        
        # Learning parameters
        self.epsilon = 0.15  # Higher exploration for mediator
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        
    def initialize_networks(self, observation_space, action_space):
        """Initialize neural networks"""
        self.policy_network = MediatorPolicyNetwork(observation_space, action_space)
        self.target_network = MediatorPolicyNetwork(observation_space, action_space)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get mediator action focusing on dispute resolution and negotiation facilitation"""
        
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
        
        # Mediation-specific actions
        mediation_strategy_probs = torch.softmax(outputs['mediation_strategy_logits'], dim=-1)
        action['mediation_strategy'] = torch.multinomial(mediation_strategy_probs, 1).item()
        
        fee_structure_probs = torch.softmax(outputs['fee_structure_logits'], dim=-1)
        action['fee_structure'] = torch.multinomial(fee_structure_probs, 1).item()
        
        conflict_priority_probs = torch.softmax(outputs['conflict_priority_logits'], dim=-1)
        action['conflict_priority'] = torch.multinomial(conflict_priority_probs, 1).item()
        
        negotiation_style_probs = torch.softmax(outputs['negotiation_style_logits'], dim=-1)
        action['negotiation_style'] = torch.multinomial(negotiation_style_probs, 1).item()
        
        # Trade actions (mediators can trade but focus on facilitation)
        if random.random() < self.epsilon:
            # Exploration
            action['trade_resource_type'] = random.randint(0, len(ResourceType) - 1)
            action['trade_quantity'] = [random.uniform(1, 50)]
            action['trade_price'] = [random.uniform(50, 1000)]
            action['trade_target'] = random.randint(0, 9)
            action['trade_action_type'] = random.choice([0, 1, 2])  # Buy, sell, or no action
        else:
            # Exploitation - focus on facilitation trades
            trade_resource_probs = torch.softmax(outputs['trade_resource_logits'], dim=-1)
            action['trade_resource_type'] = torch.multinomial(trade_resource_probs, 1).item()
            action['trade_quantity'] = outputs['trade_quantity'].cpu().numpy()
            action['trade_price'] = outputs['trade_price'].cpu().numpy()
            
            trade_target_probs = torch.softmax(outputs['trade_target_logits'], dim=-1)
            action['trade_target'] = torch.multinomial(trade_target_probs, 1).item()
            
            trade_action_probs = torch.softmax(outputs['trade_action_logits'], dim=-1)
            action['trade_action_type'] = torch.multinomial(trade_action_probs, 1).item()
        
        # Apply mediator-specific logic
        action = self._apply_mediation_heuristics(action, observation)
        
        # Communication actions (very important for mediators)
        comm_enabled_probs = torch.softmax(outputs['comm_enabled_logits'], dim=-1)
        action['comm_enabled'] = torch.multinomial(comm_enabled_probs, 1).item()
        
        # Mediators communicate frequently
        if action['comm_enabled'] or random.random() < 0.3:
            action['comm_enabled'] = 1
            comm_message_probs = torch.softmax(outputs['comm_message_logits'], dim=-1)
            action['comm_message_type'] = torch.multinomial(comm_message_probs, 1).item()
            
            comm_target_probs = torch.softmax(outputs['comm_target_logits'], dim=-1)
            action['comm_target'] = torch.multinomial(comm_target_probs, 1).item()
        else:
            action['comm_message_type'] = 0
            action['comm_target'] = 0
        
        # Alliance actions (mediators often form professional networks)
        alliance_action_probs = torch.softmax(outputs['alliance_action_logits'], dim=-1)
        action['alliance_action'] = torch.multinomial(alliance_action_probs, 1).item()
        
        alliance_target_probs = torch.softmax(outputs['alliance_target_logits'], dim=-1)
        action['alliance_target'] = torch.multinomial(alliance_target_probs, 1).item()
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return action
    
    def _apply_mediation_heuristics(self, action: Dict[str, Any], observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Apply mediator-specific heuristics"""
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        
        # Mediators prefer smaller, facilitation-focused trades
        if action['trade_action_type'] in [0, 1]:  # Buy or sell
            # Limit trade size to facilitate rather than dominate
            current_quantity = action['trade_quantity'][0]
            action['trade_quantity'] = [min(current_quantity, 25)]  # Max 25 units
            
            # Adjust pricing to be fair (close to market price)
            resource_idx = action['trade_resource_type']
            if resource_idx < len(observation['market_prices']):
                market_price = observation['market_prices'][resource_idx]
                # Price within 5% of market price
                fair_price_range = market_price * 0.1
                min_price = market_price - fair_price_range
                max_price = market_price + fair_price_range
                action['trade_price'] = [np.clip(action['trade_price'][0], min_price, max_price)]
        
        # Prioritize communication when there are disputes or negotiations
        if len(self.dispute_queue) > 0 or len(self.active_mediations) > 0:
            action['comm_enabled'] = 1
            # Use mediation-specific message types
            action['comm_message_type'] = random.choice([10, 7, 8])  # Mediation request, info request, info share
        
        return action
    
    def _get_heuristic_action(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback heuristic action for mediator"""
        
        action = {
            'trade_resource_type': 0,
            'trade_quantity': [0],
            'trade_price': [0],
            'trade_target': 0,
            'trade_action_type': 2,  # Usually no direct trading
            'comm_enabled': 1,       # Always communicate
            'comm_message_type': 7,  # Information request
            'comm_target': random.randint(0, 9),
            'alliance_action': 0,
            'alliance_target': 0,
            'mediation_strategy': 0,  # Compromise
            'fee_structure': 1,       # Medium fee
            'conflict_priority': 2,   # Medium priority
            'negotiation_style': 1    # Balanced style
        }
        
        # If we have disputes to handle, prioritize mediation
        if self.dispute_queue:
            action['comm_message_type'] = 10  # Mediation request
            # Target the first disputing party
            if self.dispute_queue[0]:
                action['comm_target'] = min(9, hash(self.dispute_queue[0]['parties'][0]) % 10)
        
        # Occasionally make small facilitation trades
        if random.random() < 0.2:
            portfolio = observation['portfolio']
            market_prices = observation['market_prices']
            
            # Find resource with moderate inventory
            for i, (quantity, price) in enumerate(zip(portfolio, market_prices)):
                if 10 < quantity < 50:  # Moderate inventory
                    action['trade_resource_type'] = i
                    action['trade_quantity'] = [min(10, quantity * 0.2)]
                    action['trade_price'] = [price * 1.05]  # Small markup
                    action['trade_action_type'] = 1  # Sell
                    break
        
        return action
    
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update mediator policy focusing on successful dispute resolution"""
        
        if self.policy_network is None or len(experiences['observations']) < 24:
            return {'loss': 0.0}
        
        batch_size = min(24, len(experiences['observations']))
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
        
        # Mediation-specific losses
        for i, (action, advantage) in enumerate(zip(batch_actions, advantages)):
            # Mediation strategy loss
            if 'mediation_strategy' in action:
                strategy_logits = outputs['mediation_strategy_logits'][i]
                strategy_target = torch.LongTensor([action['mediation_strategy']])
                strategy_loss = nn.CrossEntropyLoss()(strategy_logits.unsqueeze(0), strategy_target)
                total_loss += strategy_loss * advantage * 0.3
            
            # Communication loss (encourage communication)
            if action.get('comm_enabled', 0):
                comm_reward_bonus = 0.1
                total_loss -= comm_reward_bonus * advantage
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % 150 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item()}
    
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate mediator-specific message responses"""
        
        if message.message_type == MessageType.DISPUTE:
            # Handle dispute by offering mediation
            dispute_id = f"dispute_{self.step_count}_{message.sender}"
            self.dispute_queue.append({
                'id': dispute_id,
                'parties': [message.sender, message.content.get('other_party', 'unknown')],
                'issue': message.content.get('issue', 'trade_disagreement'),
                'timestamp': self.step_count
            })
            
            return self.generate_message(
                message.sender,
                MessageType.MEDIATION_REQUEST,
                {
                    'dispute_id': dispute_id,
                    'mediation_fee': self._calculate_mediation_fee(message.content),
                    'estimated_resolution_time': self.dispute_resolution_time,
                    'success_rate': self.success_rate
                }
            )
        
        elif message.message_type == MessageType.MEDIATION_REQUEST:
            # Someone is requesting our mediation services
            mediation_id = f"mediation_{self.step_count}_{message.sender}"
            fee = self._calculate_mediation_fee(message.content)
            
            self.active_mediations[mediation_id] = {
                'client': message.sender,
                'fee': fee,
                'start_time': self.step_count,
                'status': 'negotiating_terms'
            }
            
            return self.generate_message(
                message.sender,
                MessageType.ACCEPT,
                {
                    'mediation_id': mediation_id,
                    'fee': fee,
                    'terms': {
                        'confidentiality': True,
                        'binding_resolution': False,
                        'payment_terms': 'success_based'
                    }
                }
            )
        
        elif message.message_type == MessageType.OFFER:
            # Someone is making a trade offer - we can facilitate
            content = message.content
            
            # Analyze the offer and suggest improvements
            analysis = self._analyze_trade_offer(content)
            
            return self.generate_message(
                message.sender,
                MessageType.INFORMATION_SHARE,
                {
                    'offer_analysis': analysis,
                    'mediation_available': True,
                    'suggested_improvements': self._suggest_offer_improvements(content)
                }
            )
        
        elif message.message_type == MessageType.INFORMATION_REQUEST:
            # Provide mediation-related information
            info_type = message.content.get('info_type', 'mediation_services')
            
            if info_type == 'mediation_services':
                return self.generate_message(
                    message.sender,
                    MessageType.INFORMATION_SHARE,
                    {
                        'services': ['dispute_resolution', 'negotiation_facilitation', 'contract_review'],
                        'base_fee': self.base_mediation_fee,
                        'success_rate': self.success_rate,
                        'average_resolution_time': self.dispute_resolution_time,
                        'client_testimonials': len(self.repeat_clients)
                    }
                )
            elif info_type == 'market_fairness':
                return self.generate_message(
                    message.sender,
                    MessageType.INFORMATION_SHARE,
                    {
                        'fair_price_ranges': self._calculate_fair_price_ranges(),
                        'market_sentiment': self._assess_market_sentiment(),
                        'negotiation_tips': self._provide_negotiation_tips()
                    }
                )
        
        return None
    
    def _calculate_mediation_fee(self, case_details: Dict[str, Any]) -> float:
        """Calculate mediation fee based on case complexity and client history"""
        
        base_fee = self.base_mediation_fee
        
        # Adjust for case complexity
        complexity_multiplier = 1.0
        if 'trade_value' in case_details:
            trade_value = case_details['trade_value']
            if trade_value > 1000:
                complexity_multiplier = 1.5
            elif trade_value > 5000:
                complexity_multiplier = 2.0
        
        # Adjust for client history
        client_id = case_details.get('client_id')
        if client_id and client_id in self.repeat_clients:
            complexity_multiplier *= 0.9  # 10% discount for repeat clients
        
        # Adjust for our reputation
        reputation_multiplier = 0.8 + (self.reputation * 0.4)
        
        final_fee = base_fee * complexity_multiplier * reputation_multiplier
        return round(final_fee, 2)
    
    def _analyze_trade_offer(self, offer_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a trade offer for fairness and potential issues"""
        
        analysis = {
            'fairness_score': 0.5,
            'potential_issues': [],
            'strengths': [],
            'overall_assessment': 'neutral'
        }
        
        # Analyze price fairness
        offered_price = offer_details.get('price', 0)
        market_price = offer_details.get('market_price', offered_price)
        
        if market_price > 0:
            price_ratio = offered_price / market_price
            if 0.9 <= price_ratio <= 1.1:
                analysis['strengths'].append('fair_pricing')
                analysis['fairness_score'] += 0.2
            elif price_ratio < 0.8 or price_ratio > 1.3:
                analysis['potential_issues'].append('extreme_pricing')
                analysis['fairness_score'] -= 0.2
        
        # Analyze quantity reasonableness
        quantity = offer_details.get('quantity', 0)
        if quantity > 0:
            if quantity <= 100:
                analysis['strengths'].append('reasonable_quantity')
                analysis['fairness_score'] += 0.1
            elif quantity > 500:
                analysis['potential_issues'].append('excessive_quantity')
                analysis['fairness_score'] -= 0.1
        
        # Overall assessment
        if analysis['fairness_score'] > 0.7:
            analysis['overall_assessment'] = 'favorable'
        elif analysis['fairness_score'] < 0.3:
            analysis['overall_assessment'] = 'problematic'
        
        return analysis
    
    def _suggest_offer_improvements(self, offer_details: Dict[str, Any]) -> List[str]:
        """Suggest improvements to a trade offer"""
        
        suggestions = []
        
        offered_price = offer_details.get('price', 0)
        market_price = offer_details.get('market_price', offered_price)
        
        if market_price > 0:
            price_ratio = offered_price / market_price
            if price_ratio > 1.2:
                suggestions.append(f"Consider reducing price by {((price_ratio - 1.1) * 100):.1f}% to be more competitive")
            elif price_ratio < 0.8:
                suggestions.append("Price may be too low - consider increasing to reflect fair value")
        
        quantity = offer_details.get('quantity', 0)
        if quantity > 200:
            suggestions.append("Consider breaking large orders into smaller batches")
        
        if not offer_details.get('quality_guarantee'):
            suggestions.append("Add quality guarantee to increase buyer confidence")
        
        return suggestions
    
    def _calculate_fair_price_ranges(self) -> Dict[str, Dict[str, float]]:
        """Calculate fair price ranges for different resources"""
        
        ranges = {}
        for resource_type in ResourceType:
            ranges[resource_type.value] = {
                'min_fair': 50.0,   # Would be calculated from market data
                'max_fair': 150.0,  # Would be calculated from market data
                'recommended': 100.0
            }
        
        return ranges
    
    def _assess_market_sentiment(self) -> str:
        """Assess overall market sentiment"""
        
        # This would analyze recent trades, price trends, etc.
        sentiments = ['bullish', 'bearish', 'neutral', 'volatile']
        return random.choice(sentiments)  # Simplified for now
    
    def _provide_negotiation_tips(self) -> List[str]:
        """Provide general negotiation tips"""
        
        tips = [
            "Focus on mutual benefits rather than zero-sum outcomes",
            "Establish trust through transparent communication",
            "Be prepared to walk away if terms are unfair",
            "Consider long-term relationship value, not just immediate profit",
            "Document all agreements clearly to avoid future disputes"
        ]
        
        return random.sample(tips, 3)  # Return 3 random tips
    
    def complete_mediation(self, mediation_id: str, outcome: str, client_satisfaction: float):
        """Complete a mediation and update statistics"""
        
        if mediation_id in self.active_mediations:
            mediation = self.active_mediations[mediation_id]
            mediation['status'] = 'completed'
            mediation['outcome'] = outcome
            mediation['end_time'] = self.step_count
            mediation['client_satisfaction'] = client_satisfaction
            
            # Update statistics
            self.mediation_history.append(mediation)
            
            client_id = mediation['client']
            self.client_satisfaction[client_id] = client_satisfaction
            
            if client_satisfaction > 0.7:
                self.repeat_clients.add(client_id)
            
            # Update success rate
            successful_mediations = len([m for m in self.mediation_history if m.get('outcome') == 'resolved'])
            self.success_rate = successful_mediations / len(self.mediation_history) if self.mediation_history else 0.7
            
            # Earn fee
            if outcome == 'resolved':
                fee_earned = mediation['fee']
                self.cash += fee_earned
                self.total_profit += fee_earned
            
            del self.active_mediations[mediation_id]
    
    def get_mediation_report(self) -> Dict[str, Any]:
        """Generate mediation performance report"""
        
        return {
            'active_mediations': len(self.active_mediations),
            'completed_mediations': len(self.mediation_history),
            'success_rate': self.success_rate,
            'average_satisfaction': np.mean(list(self.client_satisfaction.values())) if self.client_satisfaction else 0.0,
            'repeat_clients': len(self.repeat_clients),
            'total_mediation_revenue': sum(m.get('fee', 0) for m in self.mediation_history if m.get('outcome') == 'resolved'),
            'average_resolution_time': np.mean([m['end_time'] - m['start_time'] for m in self.mediation_history if 'end_time' in m]) if self.mediation_history else 0,
            'pending_disputes': len(self.dispute_queue)
        }