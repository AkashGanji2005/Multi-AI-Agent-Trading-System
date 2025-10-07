"""
Regulator Agent Implementation
Ensures fair trading practices, market stability, and regulatory compliance
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

class RegulatorPolicyNetwork(nn.Module):
    """Neural network for regulator decision making"""
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = MarketplaceFeatureExtractor(observation_space, hidden_dim)
        
        # Regulator-specific heads
        self.market_intervention_head = nn.Linear(hidden_dim, 4)  # No action, price ceiling, price floor, trade halt
        self.penalty_severity_head = nn.Linear(hidden_dim, 3)    # Light, moderate, severe
        self.investigation_priority_head = nn.Linear(hidden_dim, 10)  # Which agent to investigate
        
        # Standard action heads (limited for regulator)
        self.comm_message_head = nn.Linear(hidden_dim, len(MessageType))
        self.comm_target_head = nn.Linear(hidden_dim, 10)
        self.comm_enabled_head = nn.Linear(hidden_dim, 2)
        
        # Regulators don't typically form alliances or trade
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, observations):
        features = self.feature_extractor(observations)
        
        # Regulator-specific outputs
        market_intervention_logits = self.market_intervention_head(features)
        penalty_severity_logits = self.penalty_severity_head(features)
        investigation_priority_logits = self.investigation_priority_head(features)
        
        # Communication outputs
        comm_message_logits = self.comm_message_head(features)
        comm_target_logits = self.comm_target_head(features)
        comm_enabled_logits = self.comm_enabled_head(features)
        
        value = self.value_head(features)
        
        return {
            'market_intervention_logits': market_intervention_logits,
            'penalty_severity_logits': penalty_severity_logits,
            'investigation_priority_logits': investigation_priority_logits,
            'comm_message_logits': comm_message_logits,
            'comm_target_logits': comm_target_logits,
            'comm_enabled_logits': comm_enabled_logits,
            'value': value
        }

class RegulatorAgent(BaseAgent):
    """
    Regulator agent that ensures fair trading and market stability
    Monitors for violations, applies penalties, and intervenes when necessary
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.REGULATOR, **kwargs)
        
        # Regulator-specific parameters
        self.fairness_threshold = 0.7  # Minimum fairness score to avoid intervention
        self.price_manipulation_threshold = 0.3  # Maximum price deviation before investigation
        self.monopoly_threshold = 0.6  # Market share threshold for monopoly concerns
        self.investigation_cooldown = 10  # Steps between investigations of same agent
        
        # Regulatory tools
        self.penalty_rates = {
            'light': 0.05,    # 5% of cash
            'moderate': 0.15, # 15% of cash  
            'severe': 0.30    # 30% of cash
        }
        
        self.intervention_powers = {
            'price_ceiling': True,
            'price_floor': True,
            'trade_halt': True,
            'asset_freeze': True,
            'market_ban': True
        }
        
        # Market monitoring
        self.market_history = []
        self.agent_violations = defaultdict(list)
        self.investigation_history = defaultdict(list)
        self.active_investigations = set()
        self.market_alerts = []
        
        # Fairness metrics
        self.gini_coefficient_history = []
        self.price_volatility_history = {rt: [] for rt in ResourceType}
        self.trade_concentration_scores = {}
        
        # Networks
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        
        # Learning parameters
        self.epsilon = 0.05  # Lower exploration for regulator
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        
    def initialize_networks(self, observation_space, action_space):
        """Initialize neural networks"""
        self.policy_network = RegulatorPolicyNetwork(observation_space, action_space)
        self.target_network = RegulatorPolicyNetwork(observation_space, action_space)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get regulatory action based on market analysis"""
        
        # Always analyze market conditions first
        market_alerts = self._analyze_market_conditions(observation)
        
        if self.policy_network is None:
            return self._get_heuristic_regulatory_action(observation, market_alerts)
        
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
        
        # Regulatory actions based on market conditions
        if market_alerts or random.random() < self.epsilon:
            # Take action if there are alerts or exploring
            intervention_probs = torch.softmax(outputs['market_intervention_logits'], dim=-1)
            intervention_action = torch.multinomial(intervention_probs, 1).item()
            
            penalty_probs = torch.softmax(outputs['penalty_severity_logits'], dim=-1)
            penalty_severity = torch.multinomial(penalty_probs, 1).item()
            
            investigation_probs = torch.softmax(outputs['investigation_priority_logits'], dim=-1)
            investigation_target = torch.multinomial(investigation_probs, 1).item()
            
            action.update({
                'market_intervention': intervention_action,
                'penalty_severity': penalty_severity,
                'investigation_target': investigation_target
            })
        else:
            # No intervention needed
            action.update({
                'market_intervention': 0,  # No action
                'penalty_severity': 0,     # Light
                'investigation_target': 0
            })
        
        # Regulators don't trade but may communicate
        action.update({
            'trade_resource_type': 0,
            'trade_quantity': [0],
            'trade_price': [0],
            'trade_target': 0,
            'trade_action_type': 2,  # No trade action
            'alliance_action': 0,     # No alliance action
            'alliance_target': 0
        })
        
        # Communication (warnings, announcements, information requests)
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
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return action
    
    def _analyze_market_conditions(self, observation: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze market for potential violations and instabilities"""
        
        alerts = []
        
        market_prices = observation['market_prices']
        other_agents_reputation = observation['other_agents_reputation']
        time_step = observation['time_step'][0]
        
        # Check for price manipulation
        for i, price in enumerate(market_prices):
            resource_type = list(ResourceType)[i]
            
            # Store price history
            if resource_type not in self.price_volatility_history:
                self.price_volatility_history[resource_type] = []
            self.price_volatility_history[resource_type].append(price)
            
            # Keep only recent history
            if len(self.price_volatility_history[resource_type]) > 20:
                self.price_volatility_history[resource_type] = self.price_volatility_history[resource_type][-20:]
            
            # Check for excessive volatility
            if len(self.price_volatility_history[resource_type]) >= 5:
                recent_prices = self.price_volatility_history[resource_type][-5:]
                volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
                
                if volatility > self.price_manipulation_threshold:
                    alerts.append({
                        'type': 'price_manipulation',
                        'resource': resource_type,
                        'volatility': volatility,
                        'severity': 'high' if volatility > 0.5 else 'medium'
                    })
        
        # Check for market concentration (potential monopolies)
        # This would require more detailed market share data in a real implementation
        low_reputation_agents = sum(1 for rep in other_agents_reputation if rep < 0.3)
        if low_reputation_agents > len(other_agents_reputation) * 0.3:
            alerts.append({
                'type': 'market_quality',
                'issue': 'low_reputation_agents',
                'count': low_reputation_agents,
                'severity': 'medium'
            })
        
        # Check for market fairness (Gini coefficient approximation)
        if len(other_agents_reputation) > 0:
            sorted_reps = sorted(other_agents_reputation)
            n = len(sorted_reps)
            cumsum = np.cumsum(sorted_reps)
            gini = (n + 1 - 2 * sum((n + 1 - i) * rep for i, rep in enumerate(sorted_reps, 1))) / (n * sum(sorted_reps))
            
            self.gini_coefficient_history.append(gini)
            
            if gini > 0.6:  # High inequality
                alerts.append({
                    'type': 'market_inequality',
                    'gini_coefficient': gini,
                    'severity': 'high' if gini > 0.8 else 'medium'
                })
        
        self.market_alerts = alerts
        return alerts
    
    def _get_heuristic_regulatory_action(self, observation: Dict[str, np.ndarray], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback heuristic regulatory action"""
        
        action = {
            'trade_resource_type': 0,
            'trade_quantity': [0],
            'trade_price': [0],
            'trade_target': 0,
            'trade_action_type': 2,  # No trade
            'comm_enabled': 0,
            'comm_message_type': 0,
            'comm_target': 0,
            'alliance_action': 0,
            'alliance_target': 0,
            'market_intervention': 0,
            'penalty_severity': 0,
            'investigation_target': 0
        }
        
        # React to alerts
        if alerts:
            high_severity_alerts = [a for a in alerts if a.get('severity') == 'high']
            
            if high_severity_alerts:
                # Take strong action for high severity issues
                action['market_intervention'] = 2  # Price intervention
                action['penalty_severity'] = 2    # Severe penalty
                action['comm_enabled'] = 1
                action['comm_message_type'] = 9   # Dispute/warning message
                action['comm_target'] = random.randint(0, 9)
            elif alerts:
                # Moderate action for medium severity issues
                action['market_intervention'] = 1  # Light intervention
                action['penalty_severity'] = 1    # Moderate penalty
                action['comm_enabled'] = 1
                action['comm_message_type'] = 7   # Information request
                action['comm_target'] = random.randint(0, 9)
        
        return action
    
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update regulator policy"""
        
        if self.policy_network is None or len(experiences['observations']) < 16:
            return {'loss': 0.0}
        
        batch_size = min(16, len(experiences['observations']))  # Smaller batch for regulator
        indices = random.sample(range(len(experiences['observations'])), batch_size)
        
        # Prepare batch
        batch_obs = {}
        for key in experiences['observations'][0].keys():
            batch_obs[key] = torch.FloatTensor([experiences['observations'][i][key] for i in indices])
        
        batch_rewards = torch.FloatTensor([experiences['rewards'][i] for i in indices])
        
        # Forward pass
        outputs = self.policy_network(batch_obs)
        
        # Calculate losses
        total_loss = 0.0
        
        # Value loss
        predicted_values = outputs['value'].squeeze()
        value_loss = nn.MSELoss()(predicted_values, batch_rewards)
        total_loss += value_loss
        
        # Regulatory action losses (encourage appropriate interventions)
        # This would be more sophisticated in practice, considering market outcomes
        for i in indices:
            # Reward for maintaining market stability
            market_stability_reward = experiences['rewards'][i]
            if market_stability_reward > 0:
                total_loss -= 0.1 * market_stability_reward  # Negative loss = positive reward
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % 200 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item()}
    
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate regulator-specific message responses"""
        
        if message.message_type == MessageType.DISPUTE:
            # Handle dispute resolution
            return self.generate_message(
                message.sender,
                MessageType.MEDIATION_REQUEST,
                {
                    'case_id': f"case_{self.step_count}_{message.sender}",
                    'investigation_started': True,
                    'expected_resolution_time': 5
                }
            )
        
        elif message.message_type == MessageType.INFORMATION_REQUEST:
            # Provide regulatory information
            info_type = message.content.get('info_type', 'regulations')
            
            if info_type == 'regulations':
                return self.generate_message(
                    message.sender,
                    MessageType.INFORMATION_SHARE,
                    {
                        'max_price_deviation': self.price_manipulation_threshold,
                        'penalty_rates': self.penalty_rates,
                        'current_violations': len(self.agent_violations[message.sender])
                    }
                )
            elif info_type == 'market_status':
                return self.generate_message(
                    message.sender,
                    MessageType.INFORMATION_SHARE,
                    {
                        'market_alerts': len(self.market_alerts),
                        'active_investigations': len(self.active_investigations),
                        'market_stability': 'stable' if not self.market_alerts else 'unstable'
                    }
                )
        
        return None
    
    def apply_penalty(self, agent_id: str, violation_type: str, severity: str) -> float:
        """Apply penalty to an agent"""
        
        penalty_rate = self.penalty_rates.get(severity, self.penalty_rates['light'])
        
        # Record violation
        self.agent_violations[agent_id].append({
            'type': violation_type,
            'severity': severity,
            'timestamp': self.step_count,
            'penalty_rate': penalty_rate
        })
        
        # Calculate penalty amount (this would be applied by the environment)
        penalty_amount = penalty_rate  # Percentage of agent's cash
        
        return penalty_amount
    
    def start_investigation(self, agent_id: str, reason: str) -> str:
        """Start investigation of an agent"""
        
        investigation_id = f"inv_{self.step_count}_{agent_id}"
        
        self.active_investigations.add(investigation_id)
        self.investigation_history[agent_id].append({
            'investigation_id': investigation_id,
            'reason': reason,
            'start_time': self.step_count,
            'status': 'active'
        })
        
        return investigation_id
    
    def close_investigation(self, investigation_id: str, outcome: str) -> Dict[str, Any]:
        """Close an investigation"""
        
        self.active_investigations.discard(investigation_id)
        
        # Find and update investigation record
        for agent_investigations in self.investigation_history.values():
            for inv in agent_investigations:
                if inv['investigation_id'] == investigation_id:
                    inv['status'] = 'closed'
                    inv['outcome'] = outcome
                    inv['end_time'] = self.step_count
                    return inv
        
        return {}
    
    def calculate_market_health_score(self, observation: Dict[str, np.ndarray]) -> float:
        """Calculate overall market health score"""
        
        score = 1.0
        
        # Price stability component
        price_stability = 0.0
        for resource_type, prices in self.price_volatility_history.items():
            if len(prices) >= 5:
                volatility = np.std(prices[-5:]) / np.mean(prices[-5:]) if np.mean(prices[-5:]) > 0 else 0
                price_stability += max(0, 1 - volatility)
        
        if self.price_volatility_history:
            price_stability /= len(self.price_volatility_history)
            score *= price_stability
        
        # Market fairness component (based on reputation distribution)
        other_agents_reputation = observation['other_agents_reputation']
        if len(other_agents_reputation) > 0:
            reputation_fairness = 1 - np.std(other_agents_reputation)
            score *= max(0, reputation_fairness)
        
        # Violation rate component
        total_violations = sum(len(violations) for violations in self.agent_violations.values())
        violation_penalty = min(0.5, total_violations / 100)  # Cap penalty at 50%
        score *= (1 - violation_penalty)
        
        return max(0, min(1, score))
    
    def get_regulatory_report(self) -> Dict[str, Any]:
        """Generate regulatory report"""
        
        return {
            'active_investigations': len(self.active_investigations),
            'total_violations': sum(len(violations) for violations in self.agent_violations.values()),
            'market_alerts': len(self.market_alerts),
            'recent_alerts': self.market_alerts[-5:] if self.market_alerts else [],
            'market_health_score': self.calculate_market_health_score({}),  # Would pass current observation
            'penalty_revenue': sum(
                violation['penalty_rate'] for violations in self.agent_violations.values()
                for violation in violations if violation['timestamp'] > self.step_count - 100
            ),
            'intervention_count': len([alert for alert in self.market_alerts 
                                     if alert.get('severity') in ['high', 'medium']])
        }