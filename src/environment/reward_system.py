"""
Reward System for Multi-Agent Marketplace
Implements sophisticated reward mechanisms with penalties for bad behavior
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from .marketplace import AgentType, ResourceType, Trade, Message, MessageType

class RewardComponent(Enum):
    TRADING_PERFORMANCE = "trading_performance"
    MARKET_EFFICIENCY = "market_efficiency"
    COOPERATION = "cooperation"
    REPUTATION = "reputation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    INNOVATION = "innovation"
    RISK_MANAGEMENT = "risk_management"

class PenaltyType(Enum):
    MARKET_MANIPULATION = "market_manipulation"
    UNFAIR_PRICING = "unfair_pricing"
    BREACH_OF_CONTRACT = "breach_of_contract"
    SPAM_COMMUNICATION = "spam_communication"
    MONOPOLISTIC_BEHAVIOR = "monopolistic_behavior"
    REGULATORY_VIOLATION = "regulatory_violation"
    POOR_QUALITY = "poor_quality"

@dataclass
class RewardConfig:
    """Configuration for reward system"""
    
    # Component weights (should sum to 1.0)
    trading_performance_weight: float = 0.4
    market_efficiency_weight: float = 0.2
    cooperation_weight: float = 0.15
    reputation_weight: float = 0.1
    regulatory_compliance_weight: float = 0.1
    innovation_weight: float = 0.03
    risk_management_weight: float = 0.02
    
    # Penalty multipliers
    penalty_multipliers: Dict[PenaltyType, float] = None
    
    # Reward scaling
    base_reward_scale: float = 1.0
    max_reward: float = 10.0
    min_reward: float = -10.0
    
    # Time-based factors
    discount_factor: float = 0.99
    long_term_bonus: float = 0.1
    
    def __post_init__(self):
        if self.penalty_multipliers is None:
            self.penalty_multipliers = {
                PenaltyType.MARKET_MANIPULATION: 2.0,
                PenaltyType.UNFAIR_PRICING: 1.5,
                PenaltyType.BREACH_OF_CONTRACT: 2.5,
                PenaltyType.SPAM_COMMUNICATION: 1.2,
                PenaltyType.MONOPOLISTIC_BEHAVIOR: 3.0,
                PenaltyType.REGULATORY_VIOLATION: 2.0,
                PenaltyType.POOR_QUALITY: 1.3
            }

class RewardSystem:
    """
    Comprehensive reward system that evaluates agent performance across multiple dimensions
    Includes sophisticated penalty mechanisms and agent-specific objectives
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        
        # Historical tracking for reward calculation
        self.agent_histories: Dict[str, Dict[str, List[Any]]] = {}
        self.market_history: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines: Dict[AgentType, Dict[str, float]] = {
            AgentType.BUYER: {
                'average_price_paid': 100.0,
                'quality_obtained': 0.7,
                'trade_success_rate': 0.6
            },
            AgentType.SELLER: {
                'average_profit_margin': 0.2,
                'inventory_turnover': 0.7,
                'customer_satisfaction': 0.6
            },
            AgentType.REGULATOR: {
                'market_stability': 0.8,
                'violation_detection_rate': 0.7,
                'intervention_success_rate': 0.6
            },
            AgentType.MEDIATOR: {
                'dispute_resolution_rate': 0.7,
                'client_satisfaction': 0.6,
                'mediation_efficiency': 0.6
            },
            AgentType.SPECULATOR: {
                'profit_rate': 0.3,
                'risk_adjusted_return': 0.2,
                'market_impact': 0.1
            }
        }
        
        # Penalty tracking
        self.penalty_history: Dict[str, List[Dict[str, Any]]] = {}
        self.violation_counts: Dict[str, Dict[PenaltyType, int]] = {}
        
        # Market efficiency metrics
        self.market_efficiency_history: List[float] = []
        self.price_volatility_history: Dict[ResourceType, List[float]] = {}
        
    def calculate_reward(self, 
                        agent_id: str,
                        agent_type: AgentType,
                        action: Dict[str, Any],
                        observation: Dict[str, np.ndarray],
                        market_state: Dict[str, Any],
                        trade_results: List[Trade] = None,
                        messages: List[Message] = None) -> float:
        """Calculate comprehensive reward for an agent"""
        
        # Initialize agent history if not exists
        if agent_id not in self.agent_histories:
            self.agent_histories[agent_id] = {
                'rewards': [],
                'trades': [],
                'messages': [],
                'violations': [],
                'performance_metrics': []
            }
        
        # Calculate individual reward components
        reward_components = {}
        
        # 1. Trading Performance Reward
        reward_components[RewardComponent.TRADING_PERFORMANCE] = self._calculate_trading_performance_reward(
            agent_id, agent_type, action, observation, trade_results
        )
        
        # 2. Market Efficiency Contribution
        reward_components[RewardComponent.MARKET_EFFICIENCY] = self._calculate_market_efficiency_reward(
            agent_id, agent_type, market_state
        )
        
        # 3. Cooperation and Social Behavior
        reward_components[RewardComponent.COOPERATION] = self._calculate_cooperation_reward(
            agent_id, agent_type, messages, market_state
        )
        
        # 4. Reputation Impact
        reward_components[RewardComponent.REPUTATION] = self._calculate_reputation_reward(
            agent_id, observation.get('reputation', [0.5])[0]
        )
        
        # 5. Regulatory Compliance
        reward_components[RewardComponent.REGULATORY_COMPLIANCE] = self._calculate_compliance_reward(
            agent_id, agent_type, action, market_state
        )
        
        # 6. Innovation and Adaptation
        reward_components[RewardComponent.INNOVATION] = self._calculate_innovation_reward(
            agent_id, agent_type, action
        )
        
        # 7. Risk Management
        reward_components[RewardComponent.RISK_MANAGEMENT] = self._calculate_risk_management_reward(
            agent_id, agent_type, observation, action
        )
        
        # Combine reward components with weights
        total_reward = 0.0
        total_reward += reward_components[RewardComponent.TRADING_PERFORMANCE] * self.config.trading_performance_weight
        total_reward += reward_components[RewardComponent.MARKET_EFFICIENCY] * self.config.market_efficiency_weight
        total_reward += reward_components[RewardComponent.COOPERATION] * self.config.cooperation_weight
        total_reward += reward_components[RewardComponent.REPUTATION] * self.config.reputation_weight
        total_reward += reward_components[RewardComponent.REGULATORY_COMPLIANCE] * self.config.regulatory_compliance_weight
        total_reward += reward_components[RewardComponent.INNOVATION] * self.config.innovation_weight
        total_reward += reward_components[RewardComponent.RISK_MANAGEMENT] * self.config.risk_management_weight
        
        # Apply penalties
        penalties = self._calculate_penalties(agent_id, agent_type, action, observation, market_state)
        total_reward -= penalties
        
        # Apply agent-specific bonuses
        agent_bonus = self._calculate_agent_specific_bonus(agent_id, agent_type, reward_components)
        total_reward += agent_bonus
        
        # Apply time-based factors
        total_reward = self._apply_time_factors(total_reward, agent_id)
        
        # Scale and clip reward
        total_reward *= self.config.base_reward_scale
        total_reward = np.clip(total_reward, self.config.min_reward, self.config.max_reward)
        
        # Store reward history
        self.agent_histories[agent_id]['rewards'].append({
            'total_reward': total_reward,
            'components': reward_components,
            'penalties': penalties,
            'timestamp': len(self.agent_histories[agent_id]['rewards'])
        })
        
        return total_reward
    
    def _calculate_trading_performance_reward(self,
                                            agent_id: str,
                                            agent_type: AgentType,
                                            action: Dict[str, Any],
                                            observation: Dict[str, np.ndarray],
                                            trade_results: List[Trade]) -> float:
        """Calculate reward based on trading performance"""
        
        reward = 0.0
        
        if not trade_results:
            return reward
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        market_prices = observation['market_prices']
        
        for trade in trade_results:
            if trade.status == "completed":
                if agent_type == AgentType.BUYER:
                    # Reward buyers for getting good deals
                    resource_idx = list(ResourceType).index(trade.resource.type)
                    market_price = market_prices[resource_idx] if resource_idx < len(market_prices) else trade.agreed_price
                    
                    if trade.agreed_price < market_price:
                        savings = (market_price - trade.agreed_price) * trade.resource.quantity
                        reward += savings / 1000  # Normalize
                    
                    # Quality bonus
                    quality_bonus = (trade.resource.quality - 0.5) * 0.5
                    reward += quality_bonus
                    
                elif agent_type == AgentType.SELLER:
                    # Reward sellers for profitable trades
                    resource_idx = list(ResourceType).index(trade.resource.type)
                    market_price = market_prices[resource_idx] if resource_idx < len(market_prices) else trade.agreed_price
                    
                    if trade.agreed_price > market_price:
                        profit = (trade.agreed_price - market_price) * trade.resource.quantity
                        reward += profit / 1000  # Normalize
                    
                    # Volume bonus for moving inventory
                    volume_bonus = min(0.5, trade.resource.quantity / 100)
                    reward += volume_bonus
                
                elif agent_type == AgentType.SPECULATOR:
                    # Reward speculators for profitable arbitrage
                    resource_idx = list(ResourceType).index(trade.resource.type)
                    market_price = market_prices[resource_idx] if resource_idx < len(market_prices) else trade.agreed_price
                    
                    price_difference = abs(trade.agreed_price - market_price)
                    arbitrage_profit = price_difference * trade.resource.quantity
                    reward += arbitrage_profit / 1000  # Normalize
                
            elif trade.status == "failed":
                # Small penalty for failed trades
                reward -= 0.1
        
        return reward
    
    def _calculate_market_efficiency_reward(self,
                                          agent_id: str,
                                          agent_type: AgentType,
                                          market_state: Dict[str, Any]) -> float:
        """Calculate reward based on contribution to market efficiency"""
        
        reward = 0.0
        
        # Price discovery contribution
        price_volatility = market_state.get('price_volatility', 0.0)
        if price_volatility < 0.1:  # Low volatility is good
            reward += 0.2
        elif price_volatility > 0.3:  # High volatility is bad
            reward -= 0.2
        
        # Liquidity contribution
        trade_volume = market_state.get('trade_volume', 0)
        if trade_volume > 10:  # High volume improves liquidity
            reward += 0.1
        
        # Bid-ask spread improvement (for market makers)
        if agent_type in [AgentType.MEDIATOR, AgentType.SPECULATOR]:
            spread_improvement = market_state.get('spread_improvement', 0.0)
            reward += spread_improvement * 0.5
        
        # Market depth contribution
        market_depth = market_state.get('market_depth', 0.5)
        if market_depth > 0.7:
            reward += 0.1
        
        return reward
    
    def _calculate_cooperation_reward(self,
                                    agent_id: str,
                                    agent_type: AgentType,
                                    messages: List[Message],
                                    market_state: Dict[str, Any]) -> float:
        """Calculate reward based on cooperative behavior"""
        
        reward = 0.0
        
        if not messages:
            return reward
        
        # Information sharing reward
        info_sharing_count = len([m for m in messages if m.message_type == MessageType.INFORMATION_SHARE])
        reward += min(0.3, info_sharing_count * 0.1)
        
        # Successful mediation reward
        if agent_type == AgentType.MEDIATOR:
            mediation_requests = len([m for m in messages if m.message_type == MessageType.MEDIATION_REQUEST])
            reward += mediation_requests * 0.2
        
        # Alliance participation reward
        alliance_messages = len([m for m in messages if m.message_type in [MessageType.ALLIANCE_PROPOSAL, MessageType.ALLIANCE_ACCEPT]])
        reward += min(0.2, alliance_messages * 0.1)
        
        # Dispute resolution reward
        dispute_resolution = len([m for m in messages if m.message_type == MessageType.DISPUTE])
        if dispute_resolution > 0 and agent_type in [AgentType.MEDIATOR, AgentType.REGULATOR]:
            reward += dispute_resolution * 0.15
        
        # Communication quality (avoid spam)
        total_messages = len(messages)
        if total_messages > 20:  # Too many messages
            reward -= 0.3
        
        return reward
    
    def _calculate_reputation_reward(self, agent_id: str, reputation: float) -> float:
        """Calculate reward based on reputation"""
        
        # Linear reward for reputation above 0.5
        if reputation > 0.5:
            return (reputation - 0.5) * 0.4
        else:
            return (reputation - 0.5) * 0.6  # Larger penalty for low reputation
    
    def _calculate_compliance_reward(self,
                                   agent_id: str,
                                   agent_type: AgentType,
                                   action: Dict[str, Any],
                                   market_state: Dict[str, Any]) -> float:
        """Calculate reward based on regulatory compliance"""
        
        reward = 0.0
        
        # No violations bonus
        violations = market_state.get('violations', {}).get(agent_id, 0)
        if violations == 0:
            reward += 0.2
        else:
            reward -= violations * 0.1
        
        # Regulatory agent specific rewards
        if agent_type == AgentType.REGULATOR:
            market_stability = market_state.get('market_stability', 0.5)
            reward += (market_stability - 0.5) * 0.5
            
            violations_detected = market_state.get('violations_detected', 0)
            reward += violations_detected * 0.1
        
        # Fair pricing reward
        if action.get('trade_action_type') in [0, 1]:  # Buy or sell
            price = action.get('trade_price', [0])[0]
            market_price = market_state.get('current_market_prices', {}).get(
                action.get('trade_resource_type', 0), price
            )
            
            if market_price > 0:
                price_fairness = 1.0 - abs(price - market_price) / market_price
                if price_fairness > 0.9:  # Very fair pricing
                    reward += 0.1
        
        return reward
    
    def _calculate_innovation_reward(self,
                                   agent_id: str,
                                   agent_type: AgentType,
                                   action: Dict[str, Any]) -> float:
        """Calculate reward for innovative behavior"""
        
        reward = 0.0
        
        # Novel action patterns
        if agent_id in self.agent_histories:
            recent_actions = self.agent_histories[agent_id].get('actions', [])[-10:]
            if recent_actions:
                # Reward for trying different strategies
                action_diversity = len(set(str(a) for a in recent_actions)) / len(recent_actions)
                reward += action_diversity * 0.1
        
        # Creative problem solving (using mediation, alliances, etc.)
        if action.get('alliance_action', 0) > 0 or action.get('comm_enabled', 0):
            reward += 0.05
        
        return reward
    
    def _calculate_risk_management_reward(self,
                                        agent_id: str,
                                        agent_type: AgentType,
                                        observation: Dict[str, np.ndarray],
                                        action: Dict[str, Any]) -> float:
        """Calculate reward for good risk management"""
        
        reward = 0.0
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        total_portfolio_value = cash + sum(portfolio)
        
        # Cash reserve management
        cash_ratio = cash / total_portfolio_value if total_portfolio_value > 0 else 1.0
        
        if 0.1 <= cash_ratio <= 0.3:  # Good cash management
            reward += 0.1
        elif cash_ratio < 0.05:  # Too little cash (risky)
            reward -= 0.2
        elif cash_ratio > 0.5:  # Too much cash (inefficient)
            reward -= 0.1
        
        # Position sizing
        if action.get('trade_action_type') in [0, 1]:  # Buy or sell
            trade_value = action.get('trade_quantity', [0])[0] * action.get('trade_price', [0])[0]
            position_size_ratio = trade_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            if position_size_ratio > 0.5:  # Too large position
                reward -= 0.3
            elif 0.05 <= position_size_ratio <= 0.2:  # Good position sizing
                reward += 0.1
        
        # Diversification
        if len(portfolio) > 0:
            portfolio_concentration = max(portfolio) / sum(portfolio) if sum(portfolio) > 0 else 0
            if portfolio_concentration < 0.4:  # Well diversified
                reward += 0.1
            elif portfolio_concentration > 0.8:  # Too concentrated
                reward -= 0.2
        
        return reward
    
    def _calculate_penalties(self,
                           agent_id: str,
                           agent_type: AgentType,
                           action: Dict[str, Any],
                           observation: Dict[str, np.ndarray],
                           market_state: Dict[str, Any]) -> float:
        """Calculate penalties for bad behavior"""
        
        total_penalty = 0.0
        
        # Initialize penalty tracking
        if agent_id not in self.penalty_history:
            self.penalty_history[agent_id] = []
        if agent_id not in self.violation_counts:
            self.violation_counts[agent_id] = {penalty_type: 0 for penalty_type in PenaltyType}
        
        # Market manipulation detection
        manipulation_penalty = self._detect_market_manipulation(agent_id, action, market_state)
        if manipulation_penalty > 0:
            total_penalty += manipulation_penalty * self.config.penalty_multipliers[PenaltyType.MARKET_MANIPULATION]
            self.violation_counts[agent_id][PenaltyType.MARKET_MANIPULATION] += 1
        
        # Unfair pricing
        unfair_pricing_penalty = self._detect_unfair_pricing(agent_id, action, market_state)
        if unfair_pricing_penalty > 0:
            total_penalty += unfair_pricing_penalty * self.config.penalty_multipliers[PenaltyType.UNFAIR_PRICING]
            self.violation_counts[agent_id][PenaltyType.UNFAIR_PRICING] += 1
        
        # Communication spam
        spam_penalty = self._detect_communication_spam(agent_id, market_state)
        if spam_penalty > 0:
            total_penalty += spam_penalty * self.config.penalty_multipliers[PenaltyType.SPAM_COMMUNICATION]
            self.violation_counts[agent_id][PenaltyType.SPAM_COMMUNICATION] += 1
        
        # Monopolistic behavior
        monopoly_penalty = self._detect_monopolistic_behavior(agent_id, observation, market_state)
        if monopoly_penalty > 0:
            total_penalty += monopoly_penalty * self.config.penalty_multipliers[PenaltyType.MONOPOLISTIC_BEHAVIOR]
            self.violation_counts[agent_id][PenaltyType.MONOPOLISTIC_BEHAVIOR] += 1
        
        # Repeated violations escalation
        total_violations = sum(self.violation_counts[agent_id].values())
        if total_violations > 5:
            escalation_penalty = min(2.0, (total_violations - 5) * 0.2)
            total_penalty += escalation_penalty
        
        return total_penalty
    
    def _detect_market_manipulation(self,
                                  agent_id: str,
                                  action: Dict[str, Any],
                                  market_state: Dict[str, Any]) -> float:
        """Detect market manipulation attempts"""
        
        penalty = 0.0
        
        # Excessive trading frequency
        recent_trades = market_state.get('recent_trades', {}).get(agent_id, 0)
        if recent_trades > 10:  # More than 10 trades in recent period
            penalty += 0.5
        
        # Wash trading detection (buying and selling same resource repeatedly)
        if agent_id in self.agent_histories:
            recent_actions = self.agent_histories[agent_id].get('actions', [])[-5:]
            buy_sell_pattern = [a.get('trade_action_type', 2) for a in recent_actions]
            
            # Check for alternating buy-sell pattern
            if len(buy_sell_pattern) >= 4:
                alternating_count = 0
                for i in range(1, len(buy_sell_pattern)):
                    if buy_sell_pattern[i] != buy_sell_pattern[i-1] and buy_sell_pattern[i] in [0, 1]:
                        alternating_count += 1
                
                if alternating_count >= 3:  # Suspicious pattern
                    penalty += 1.0
        
        # Price manipulation (extreme pricing)
        if action.get('trade_action_type') in [0, 1]:
            price = action.get('trade_price', [0])[0]
            market_price = market_state.get('current_market_prices', {}).get(
                action.get('trade_resource_type', 0), price
            )
            
            if market_price > 0 and abs(price - market_price) / market_price > 0.5:
                penalty += 0.8  # Extreme pricing penalty
        
        return penalty
    
    def _detect_unfair_pricing(self,
                             agent_id: str,
                             action: Dict[str, Any],
                             market_state: Dict[str, Any]) -> float:
        """Detect unfair pricing practices"""
        
        penalty = 0.0
        
        if action.get('trade_action_type') not in [0, 1]:
            return penalty
        
        price = action.get('trade_price', [0])[0]
        quantity = action.get('trade_quantity', [0])[0]
        resource_type = action.get('trade_resource_type', 0)
        
        market_price = market_state.get('current_market_prices', {}).get(resource_type, price)
        
        if market_price > 0:
            price_deviation = (price - market_price) / market_price
            
            # Excessive markup/markdown
            if abs(price_deviation) > 0.3:
                penalty += min(1.0, abs(price_deviation))
            
            # Predatory pricing (selling way below cost to drive out competition)
            if price_deviation < -0.4 and quantity > 50:
                penalty += 1.5
            
            # Price gouging (excessive markup during scarcity)
            market_supply = market_state.get('market_supply', {}).get(resource_type, 100)
            if market_supply < 20 and price_deviation > 0.5:
                penalty += 2.0
        
        return penalty
    
    def _detect_communication_spam(self, agent_id: str, market_state: Dict[str, Any]) -> float:
        """Detect communication spam"""
        
        penalty = 0.0
        
        message_count = market_state.get('agent_message_counts', {}).get(agent_id, 0)
        
        # Too many messages
        if message_count > 15:
            penalty += (message_count - 15) * 0.1
        
        # Repetitive messaging
        if agent_id in self.agent_histories:
            recent_messages = self.agent_histories[agent_id].get('messages', [])[-10:]
            if len(recent_messages) > 5:
                unique_message_types = len(set(m.message_type for m in recent_messages))
                if unique_message_types <= 2:  # Very repetitive
                    penalty += 0.5
        
        return penalty
    
    def _detect_monopolistic_behavior(self,
                                    agent_id: str,
                                    observation: Dict[str, np.ndarray],
                                    market_state: Dict[str, Any]) -> float:
        """Detect monopolistic behavior"""
        
        penalty = 0.0
        
        portfolio = observation['portfolio']
        total_market_supply = market_state.get('total_market_supply', {})
        
        for i, quantity in enumerate(portfolio):
            resource_type = i
            market_supply = total_market_supply.get(resource_type, quantity + 100)
            
            if market_supply > 0:
                market_share = quantity / market_supply
                
                # High market share penalty
                if market_share > 0.6:
                    penalty += (market_share - 0.6) * 2.0
                elif market_share > 0.8:
                    penalty += 3.0  # Severe monopoly penalty
        
        return penalty
    
    def _calculate_agent_specific_bonus(self,
                                      agent_id: str,
                                      agent_type: AgentType,
                                      reward_components: Dict[RewardComponent, float]) -> float:
        """Calculate agent-type specific bonuses"""
        
        bonus = 0.0
        
        if agent_type == AgentType.BUYER:
            # Bonus for efficient purchasing
            trading_reward = reward_components[RewardComponent.TRADING_PERFORMANCE]
            if trading_reward > 0.5:
                bonus += 0.2
        
        elif agent_type == AgentType.SELLER:
            # Bonus for good customer service (cooperation + reputation)
            service_score = (reward_components[RewardComponent.COOPERATION] + 
                           reward_components[RewardComponent.REPUTATION]) / 2
            if service_score > 0.3:
                bonus += 0.15
        
        elif agent_type == AgentType.REGULATOR:
            # Bonus for maintaining market stability
            compliance_reward = reward_components[RewardComponent.REGULATORY_COMPLIANCE]
            efficiency_reward = reward_components[RewardComponent.MARKET_EFFICIENCY]
            if compliance_reward > 0.2 and efficiency_reward > 0.1:
                bonus += 0.3
        
        elif agent_type == AgentType.MEDIATOR:
            # Bonus for successful dispute resolution
            cooperation_reward = reward_components[RewardComponent.COOPERATION]
            if cooperation_reward > 0.3:
                bonus += 0.25
        
        elif agent_type == AgentType.SPECULATOR:
            # Bonus for providing liquidity while being profitable
            trading_reward = reward_components[RewardComponent.TRADING_PERFORMANCE]
            efficiency_reward = reward_components[RewardComponent.MARKET_EFFICIENCY]
            if trading_reward > 0.3 and efficiency_reward > 0.1:
                bonus += 0.2
        
        return bonus
    
    def _apply_time_factors(self, reward: float, agent_id: str) -> float:
        """Apply time-based factors to reward"""
        
        # Long-term consistency bonus
        if agent_id in self.agent_histories:
            recent_rewards = [r['total_reward'] for r in self.agent_histories[agent_id]['rewards'][-10:]]
            if len(recent_rewards) >= 5:
                reward_variance = np.var(recent_rewards)
                if reward_variance < 1.0:  # Consistent performance
                    reward += self.config.long_term_bonus
        
        return reward
    
    def get_reward_breakdown(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed reward breakdown for an agent"""
        
        if agent_id not in self.agent_histories:
            return {}
        
        history = self.agent_histories[agent_id]['rewards']
        if not history:
            return {}
        
        latest_reward = history[-1]
        
        return {
            'total_reward': latest_reward['total_reward'],
            'components': latest_reward['components'],
            'penalties': latest_reward['penalties'],
            'violation_counts': self.violation_counts.get(agent_id, {}),
            'recent_average': np.mean([r['total_reward'] for r in history[-10:]]) if len(history) >= 10 else latest_reward['total_reward'],
            'reward_trend': np.mean([r['total_reward'] for r in history[-5:]]) - np.mean([r['total_reward'] for r in history[-10:-5]]) if len(history) >= 10 else 0.0
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall reward system metrics"""
        
        total_agents = len(self.agent_histories)
        if total_agents == 0:
            return {}
        
        # Calculate system-wide metrics
        all_rewards = []
        all_violations = []
        
        for agent_id, history in self.agent_histories.items():
            if history['rewards']:
                all_rewards.extend([r['total_reward'] for r in history['rewards']])
            all_violations.append(sum(self.violation_counts.get(agent_id, {}).values()))
        
        return {
            'total_agents': total_agents,
            'average_reward': np.mean(all_rewards) if all_rewards else 0.0,
            'reward_std': np.std(all_rewards) if all_rewards else 0.0,
            'total_violations': sum(all_violations),
            'violation_rate': sum(all_violations) / total_agents if total_agents > 0 else 0.0,
            'market_efficiency_trend': np.mean(self.market_efficiency_history[-10:]) if len(self.market_efficiency_history) >= 10 else 0.0
        }