"""
Speculator Agent Implementation
Exploits market inefficiencies for profit through advanced trading strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import deque, defaultdict

from .base_agent import BaseAgent, MarketplaceFeatureExtractor
from ..environment.marketplace import AgentType, ResourceType, MessageType, Message, Trade

class SpeculatorPolicyNetwork(nn.Module):
    """Neural network for speculator decision making with advanced trading strategies"""
    
    def __init__(self, observation_space, action_space, hidden_dim=512):
        super().__init__()
        
        self.feature_extractor = MarketplaceFeatureExtractor(observation_space, hidden_dim)
        
        # Speculator-specific heads for advanced strategies
        self.market_timing_head = nn.Linear(hidden_dim, 3)      # Buy, sell, hold signals
        self.arbitrage_opportunity_head = nn.Linear(hidden_dim, len(ResourceType))  # Arbitrage scores
        self.volatility_prediction_head = nn.Linear(hidden_dim, len(ResourceType))  # Price volatility predictions
        self.momentum_strategy_head = nn.Linear(hidden_dim, len(ResourceType))      # Momentum indicators
        self.risk_assessment_head = nn.Linear(hidden_dim, 5)   # Risk levels
        
        # Position sizing and portfolio management
        self.position_size_head = nn.Linear(hidden_dim, 1)     # Position size multiplier
        self.portfolio_allocation_head = nn.Linear(hidden_dim, len(ResourceType))  # Portfolio weights
        
        # Standard action heads with enhanced capabilities
        self.trade_resource_head = nn.Linear(hidden_dim, len(ResourceType))
        self.trade_quantity_head = nn.Linear(hidden_dim, 1)
        self.trade_price_head = nn.Linear(hidden_dim, 1)
        self.trade_target_head = nn.Linear(hidden_dim, 10)
        self.trade_action_head = nn.Linear(hidden_dim, 4)
        
        # Communication for market manipulation and information gathering
        self.comm_message_head = nn.Linear(hidden_dim, len(MessageType))
        self.comm_target_head = nn.Linear(hidden_dim, 10)
        self.comm_enabled_head = nn.Linear(hidden_dim, 2)
        
        # Alliance for market coordination
        self.alliance_action_head = nn.Linear(hidden_dim, 4)
        self.alliance_target_head = nn.Linear(hidden_dim, 10)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, observations):
        features = self.feature_extractor(observations)
        
        # Speculator-specific outputs
        market_timing_logits = self.market_timing_head(features)
        arbitrage_scores = torch.sigmoid(self.arbitrage_opportunity_head(features))
        volatility_predictions = torch.sigmoid(self.volatility_prediction_head(features))
        momentum_indicators = torch.tanh(self.momentum_strategy_head(features))
        risk_assessment_logits = self.risk_assessment_head(features)
        
        # Portfolio management
        position_size = torch.sigmoid(self.position_size_head(features)) * 2.0  # 0-2x multiplier
        portfolio_weights = torch.softmax(self.portfolio_allocation_head(features), dim=-1)
        
        # Standard outputs
        trade_resource_logits = self.trade_resource_head(features)
        trade_quantity = torch.sigmoid(self.trade_quantity_head(features)) * 2000  # Higher limits for speculator
        trade_price = torch.sigmoid(self.trade_price_head(features)) * 20000
        trade_target_logits = self.trade_target_head(features)
        trade_action_logits = self.trade_action_head(features)
        
        comm_message_logits = self.comm_message_head(features)
        comm_target_logits = self.comm_target_head(features)
        comm_enabled_logits = self.comm_enabled_head(features)
        
        alliance_action_logits = self.alliance_action_head(features)
        alliance_target_logits = self.alliance_target_head(features)
        
        value = self.value_head(features)
        
        return {
            'market_timing_logits': market_timing_logits,
            'arbitrage_scores': arbitrage_scores,
            'volatility_predictions': volatility_predictions,
            'momentum_indicators': momentum_indicators,
            'risk_assessment_logits': risk_assessment_logits,
            'position_size': position_size,
            'portfolio_weights': portfolio_weights,
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

class SpeculatorAgent(BaseAgent):
    """
    Speculator agent that exploits market inefficiencies for maximum profit
    Uses advanced trading strategies, market analysis, and strategic positioning
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.SPECULATOR, **kwargs)
        
        # Speculator-specific parameters
        self.profit_target = 0.5        # Target 50% returns
        self.max_drawdown = 0.3         # Maximum acceptable loss
        self.leverage_limit = 3.0       # Maximum leverage
        self.position_holding_time = 20  # Average holding period
        
        # Trading strategies
        self.strategies = {
            'arbitrage': 0.8,           # Exploit price differences
            'momentum': 0.7,            # Follow trends
            'mean_reversion': 0.6,      # Bet on price corrections
            'volatility_trading': 0.9,  # Trade on volatility
            'market_making': 0.5        # Provide liquidity
        }
        
        # Market analysis
        self.price_history = {rt: deque(maxlen=50) for rt in ResourceType}
        self.volume_history = {rt: deque(maxlen=50) for rt in ResourceType}
        self.volatility_estimates = {rt: 0.1 for rt in ResourceType}
        self.momentum_indicators = {rt: 0.0 for rt in ResourceType}
        self.support_resistance_levels = {rt: {'support': [], 'resistance': []} for rt in ResourceType}
        
        # Portfolio tracking
        self.positions = {}  # Current positions
        self.open_orders = {}  # Pending orders
        self.trade_history = []
        self.pnl_history = []
        self.max_position_size = 500
        
        # Risk management
        self.var_limit = 0.1  # Value at Risk limit
        self.stop_loss_pct = 0.15  # Stop loss percentage
        self.take_profit_pct = 0.25  # Take profit percentage
        
        # Market manipulation detection avoidance
        self.trade_frequency_limit = 5  # Max trades per period
        self.recent_trades_count = 0
        self.manipulation_risk_score = 0.0
        
        # Information networks
        self.information_sources = set()
        self.insider_information = {}
        self.market_rumors = []
        
        # Networks
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        
        # Learning parameters
        self.epsilon = 0.2  # High exploration for finding opportunities
        self.epsilon_decay = 0.998
        self.min_epsilon = 0.05
        
    def initialize_networks(self, observation_space, action_space):
        """Initialize neural networks with larger capacity for complex strategies"""
        self.policy_network = SpeculatorPolicyNetwork(observation_space, action_space, hidden_dim=512)
        self.target_network = SpeculatorPolicyNetwork(observation_space, action_space, hidden_dim=512)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
    
    def get_action(self, observation: Dict[str, np.ndarray], legal_actions: List[int] = None) -> Dict[str, Any]:
        """Get speculator action using advanced market analysis"""
        
        # Update market analysis
        self._update_market_analysis(observation)
        
        # Check for arbitrage opportunities
        arbitrage_opportunities = self._find_arbitrage_opportunities(observation)
        
        if self.policy_network is None:
            return self._get_heuristic_action(observation, arbitrage_opportunities)
        
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
        
        # Market timing decision
        market_timing_probs = torch.softmax(outputs['market_timing_logits'], dim=-1)
        market_signal = torch.multinomial(market_timing_probs, 1).item()  # 0: buy, 1: sell, 2: hold
        
        # Risk assessment
        risk_probs = torch.softmax(outputs['risk_assessment_logits'], dim=-1)
        risk_level = torch.multinomial(risk_probs, 1).item()
        
        # Position sizing
        position_multiplier = outputs['position_size'].item()
        
        # Get trading action based on strategy
        if arbitrage_opportunities and market_signal != 2:  # Not hold
            # Execute arbitrage strategy
            action = self._get_arbitrage_action(arbitrage_opportunities, outputs)
        elif random.random() < self.epsilon:
            # Exploration
            action = self._get_exploration_action()
        else:
            # Strategic action based on network
            action = self._get_strategic_action(outputs, observation, market_signal, risk_level)
        
        # Apply risk management
        action = self._apply_risk_management(action, observation, risk_level)
        
        # Communication for information gathering and potential market influence
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
        
        # Alliance actions for coordinated strategies
        alliance_action_probs = torch.softmax(outputs['alliance_action_logits'], dim=-1)
        action['alliance_action'] = torch.multinomial(alliance_action_probs, 1).item()
        
        alliance_target_probs = torch.softmax(outputs['alliance_target_logits'], dim=-1)
        action['alliance_target'] = torch.multinomial(alliance_target_probs, 1).item()
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return action
    
    def _update_market_analysis(self, observation: Dict[str, np.ndarray]):
        """Update market analysis with new data"""
        
        market_prices = observation['market_prices']
        time_step = observation['time_step'][0]
        
        for i, price in enumerate(market_prices):
            resource_type = list(ResourceType)[i]
            
            # Update price history
            self.price_history[resource_type].append(price)
            
            # Calculate volatility
            if len(self.price_history[resource_type]) > 5:
                prices = list(self.price_history[resource_type])
                returns = [np.log(prices[j]/prices[j-1]) for j in range(1, len(prices))]
                self.volatility_estimates[resource_type] = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate momentum
            if len(self.price_history[resource_type]) > 10:
                prices = list(self.price_history[resource_type])
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                self.momentum_indicators[resource_type] = (short_ma - long_ma) / long_ma
            
            # Update support/resistance levels
            if len(self.price_history[resource_type]) > 20:
                prices = list(self.price_history[resource_type])
                local_maxima = [prices[j] for j in range(1, len(prices)-1) 
                               if prices[j] > prices[j-1] and prices[j] > prices[j+1]]
                local_minima = [prices[j] for j in range(1, len(prices)-1)
                               if prices[j] < prices[j-1] and prices[j] < prices[j+1]]
                
                if local_maxima:
                    self.support_resistance_levels[resource_type]['resistance'] = local_maxima[-3:]
                if local_minima:
                    self.support_resistance_levels[resource_type]['support'] = local_minima[-3:]
    
    def _find_arbitrage_opportunities(self, observation: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities in the market"""
        
        opportunities = []
        market_prices = observation['market_prices']
        portfolio = observation['portfolio']
        
        # Simple price discrepancy arbitrage
        for i, price in enumerate(market_prices):
            resource_type = list(ResourceType)[i]
            
            # Check if price deviates significantly from recent average
            if len(self.price_history[resource_type]) > 10:
                recent_avg = np.mean(list(self.price_history[resource_type])[-10:])
                price_deviation = abs(price - recent_avg) / recent_avg
                
                if price_deviation > 0.1:  # 10% deviation
                    opportunity = {
                        'type': 'price_reversion',
                        'resource_idx': i,
                        'resource_type': resource_type,
                        'current_price': price,
                        'expected_price': recent_avg,
                        'confidence': min(0.9, price_deviation),
                        'action': 'buy' if price < recent_avg else 'sell'
                    }
                    
                    # Check if we can execute this opportunity
                    if opportunity['action'] == 'buy' and self.cash > price * 10:
                        opportunities.append(opportunity)
                    elif opportunity['action'] == 'sell' and portfolio[i] > 10:
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _get_arbitrage_action(self, opportunities: List[Dict[str, Any]], outputs) -> Dict[str, Any]:
        """Generate action to exploit arbitrage opportunity"""
        
        best_opportunity = max(opportunities, key=lambda x: x['confidence'])
        
        action = {
            'trade_resource_type': best_opportunity['resource_idx'],
            'trade_target': random.randint(0, 9),
            'alliance_action': 0,
            'alliance_target': 0
        }
        
        if best_opportunity['action'] == 'buy':
            action['trade_action_type'] = 0  # Buy
            action['trade_quantity'] = [min(100, self.cash / best_opportunity['current_price'] * 0.5)]
            action['trade_price'] = [best_opportunity['current_price'] * 0.98]  # Slightly below market
        else:  # Sell
            action['trade_action_type'] = 1  # Sell
            available_quantity = self.portfolio.get(best_opportunity['resource_type'], 0)
            action['trade_quantity'] = [min(100, available_quantity * 0.3)]
            action['trade_price'] = [best_opportunity['current_price'] * 1.02]  # Slightly above market
        
        return action
    
    def _get_exploration_action(self) -> Dict[str, Any]:
        """Generate exploration action"""
        
        return {
            'trade_resource_type': random.randint(0, len(ResourceType) - 1),
            'trade_quantity': [random.uniform(10, 200)],
            'trade_price': [random.uniform(50, 2000)],
            'trade_target': random.randint(0, 9),
            'trade_action_type': random.choice([0, 1, 2]),  # Buy, sell, or hold
            'alliance_action': 0,
            'alliance_target': 0
        }
    
    def _get_strategic_action(self, outputs, observation, market_signal, risk_level) -> Dict[str, Any]:
        """Generate strategic action based on market analysis"""
        
        portfolio_weights = outputs['portfolio_weights'].squeeze().cpu().numpy()
        arbitrage_scores = outputs['arbitrage_scores'].squeeze().cpu().numpy()
        momentum_indicators = outputs['momentum_indicators'].squeeze().cpu().numpy()
        
        # Select resource based on combined signals
        resource_scores = []
        for i, (weight, arb_score, momentum) in enumerate(zip(portfolio_weights, arbitrage_scores, momentum_indicators)):
            resource_type = list(ResourceType)[i]
            volatility = self.volatility_estimates.get(resource_type, 0.1)
            
            # Combine signals
            score = weight * 0.3 + arb_score * 0.4 + abs(momentum) * 0.3
            # Adjust for volatility (higher volatility = more opportunity but more risk)
            score *= (1 + volatility) if risk_level > 2 else (1 - volatility * 0.5)
            
            resource_scores.append((i, score))
        
        # Select best resource
        best_resource_idx = max(resource_scores, key=lambda x: x[1])[0]
        resource_type = list(ResourceType)[best_resource_idx]
        
        action = {
            'trade_resource_type': best_resource_idx,
            'trade_target': random.randint(0, 9),
            'alliance_action': 0,
            'alliance_target': 0
        }
        
        # Determine action type based on momentum and market signal
        momentum = momentum_indicators[best_resource_idx]
        current_price = observation['market_prices'][best_resource_idx]
        
        if market_signal == 0 or (market_signal == 2 and momentum > 0.1):  # Buy signal
            action['trade_action_type'] = 0
            quantity = min(200, self.cash / current_price * 0.3)  # Risk-adjusted position size
            action['trade_quantity'] = [quantity]
            action['trade_price'] = [current_price * 0.995]  # Slightly below market
            
        elif market_signal == 1 or (market_signal == 2 and momentum < -0.1):  # Sell signal
            action['trade_action_type'] = 1
            available_quantity = observation['portfolio'][best_resource_idx]
            quantity = min(200, available_quantity * 0.4)
            action['trade_quantity'] = [quantity]
            action['trade_price'] = [current_price * 1.005]  # Slightly above market
            
        else:  # Hold
            action['trade_action_type'] = 2
            action['trade_quantity'] = [0]
            action['trade_price'] = [0]
        
        return action
    
    def _apply_risk_management(self, action: Dict[str, Any], observation: Dict[str, np.ndarray], risk_level: int) -> Dict[str, Any]:
        """Apply risk management rules to the action"""
        
        cash = observation['cash'][0]
        portfolio = observation['portfolio']
        
        # Limit position size based on risk level
        risk_multipliers = [0.5, 0.7, 1.0, 1.3, 1.5]  # Conservative to aggressive
        risk_multiplier = risk_multipliers[min(risk_level, 4)]
        
        if action['trade_action_type'] == 0:  # Buy
            max_position_value = cash * 0.2 * risk_multiplier  # Max 20% of cash per position (adjusted for risk)
            current_price = action['trade_price'][0]
            max_quantity = max_position_value / current_price if current_price > 0 else 0
            
            action['trade_quantity'] = [min(action['trade_quantity'][0], max_quantity)]
            
            # Don't trade if insufficient funds
            if cash < action['trade_quantity'][0] * current_price:
                action['trade_action_type'] = 2  # No action
                action['trade_quantity'] = [0]
        
        elif action['trade_action_type'] == 1:  # Sell
            resource_idx = action['trade_resource_type']
            available_quantity = portfolio[resource_idx] if resource_idx < len(portfolio) else 0
            
            # Don't sell more than we have
            action['trade_quantity'] = [min(action['trade_quantity'][0], available_quantity)]
            
            # Don't sell if we don't have enough inventory
            if available_quantity < action['trade_quantity'][0]:
                action['trade_action_type'] = 2  # No action
                action['trade_quantity'] = [0]
        
        # Limit trading frequency to avoid manipulation detection
        self.recent_trades_count += 1 if action['trade_action_type'] in [0, 1] else 0
        if self.recent_trades_count > self.trade_frequency_limit:
            action['trade_action_type'] = 2  # Force hold
            action['trade_quantity'] = [0]
        
        # Reset trade count periodically
        if self.step_count % 20 == 0:
            self.recent_trades_count = 0
        
        return action
    
    def _get_heuristic_action(self, observation: Dict[str, np.ndarray], arbitrage_opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback heuristic action"""
        
        action = {
            'trade_resource_type': 0,
            'trade_quantity': [0],
            'trade_price': [0],
            'trade_target': 0,
            'trade_action_type': 2,  # Hold by default
            'comm_enabled': 0,
            'comm_message_type': 0,
            'comm_target': 0,
            'alliance_action': 0,
            'alliance_target': 0
        }
        
        # Execute arbitrage if available
        if arbitrage_opportunities:
            return self._get_arbitrage_action(arbitrage_opportunities, None)
        
        # Simple momentum strategy
        market_prices = observation['market_prices']
        portfolio = observation['portfolio']
        cash = observation['cash'][0]
        
        for i, price in enumerate(market_prices):
            resource_type = list(ResourceType)[i]
            
            if len(self.price_history[resource_type]) > 5:
                recent_prices = list(self.price_history[resource_type])[-5:]
                trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                if trend > 0.05 and cash > price * 20:  # Strong uptrend, buy
                    action.update({
                        'trade_resource_type': i,
                        'trade_quantity': [min(50, cash / price * 0.1)],
                        'trade_price': [price * 0.99],
                        'trade_target': random.randint(0, 9),
                        'trade_action_type': 0
                    })
                    break
                elif trend < -0.05 and portfolio[i] > 20:  # Strong downtrend, sell
                    action.update({
                        'trade_resource_type': i,
                        'trade_quantity': [min(30, portfolio[i] * 0.3)],
                        'trade_price': [price * 1.01],
                        'trade_target': random.randint(0, 9),
                        'trade_action_type': 1
                    })
                    break
        
        # Occasionally gather market information
        if random.random() < 0.2:
            action['comm_enabled'] = 1
            action['comm_message_type'] = 7  # Information request
            action['comm_target'] = random.randint(0, 9)
        
        return action
    
    def update_policy(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update speculator policy with emphasis on profit maximization"""
        
        if self.policy_network is None or len(experiences['observations']) < 64:
            return {'loss': 0.0}
        
        batch_size = min(64, len(experiences['observations']))  # Larger batch for speculator
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
        
        # Policy losses with profit-focused weighting
        advantages = batch_rewards - predicted_values.detach()
        
        # Enhanced loss calculation for speculator strategies
        for i, (action, advantage) in enumerate(zip(batch_actions, advantages)):
            # Trading action loss with profit weighting
            if action['trade_action_type'] in [0, 1]:  # Buy or sell
                resource_logits = outputs['trade_resource_logits'][i]
                resource_target = torch.LongTensor([action['trade_resource_type']])
                resource_loss = nn.CrossEntropyLoss()(resource_logits.unsqueeze(0), resource_target)
                
                # Weight by profit potential
                profit_weight = max(0.1, 1.0 + advantage)
                total_loss += resource_loss * profit_weight
                
                # Price prediction loss
                predicted_price = outputs['trade_price'][i]
                actual_price = torch.FloatTensor([action['trade_price'][0]])
                price_loss = nn.MSELoss()(predicted_price, actual_price)
                total_loss += price_loss * 0.1
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % 80 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item()}
    
    def _generate_message_response(self, message: Message) -> Optional[Message]:
        """Generate speculator-specific message responses"""
        
        if message.message_type == MessageType.INFORMATION_REQUEST:
            # Selectively share information (or misinformation)
            info_type = message.content.get('info_type', 'market_prices')
            
            if self.should_trust_agent(message.sender, 0.7):
                # Share real information with trusted agents
                if info_type == 'market_prices':
                    return self.generate_message(
                        message.sender,
                        MessageType.INFORMATION_SHARE,
                        {
                            'price_trends': {rt.value: self.momentum_indicators[rt] for rt in ResourceType},
                            'volatility_estimates': {rt.value: self.volatility_estimates[rt] for rt in ResourceType}
                        }
                    )
                elif info_type == 'trading_opportunities':
                    return self.generate_message(
                        message.sender,
                        MessageType.INFORMATION_SHARE,
                        {
                            'high_volatility_resources': [rt.value for rt, vol in self.volatility_estimates.items() if vol > 0.2],
                            'momentum_plays': [rt.value for rt, mom in self.momentum_indicators.items() if abs(mom) > 0.1]
                        }
                    )
            else:
                # Provide misleading information to competitors
                return self.generate_message(
                    message.sender,
                    MessageType.INFORMATION_SHARE,
                    {
                        'market_sentiment': 'neutral',  # Always claim neutral
                        'recommendation': 'hold_cash'   # Discourage trading
                    }
                )
        
        elif message.message_type == MessageType.ALLIANCE_PROPOSAL:
            # Consider alliance if it provides market advantages
            proposed_purpose = message.content.get('purpose', '')
            
            if 'market_coordination' in proposed_purpose or 'information_sharing' in proposed_purpose:
                if self.cooperation_tendency > 0.4:  # Lower threshold for profitable alliances
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
        
        elif message.message_type == MessageType.OFFER:
            # Evaluate offers for arbitrage potential
            content = message.content
            offered_price = content.get('price', 0)
            resource_type = content.get('resource_type', '')
            
            if resource_type and hasattr(ResourceType, resource_type.upper()):
                rt = ResourceType[resource_type.upper()]
                expected_price = np.mean(list(self.price_history[rt])[-5:]) if self.price_history[rt] else offered_price
                
                price_difference = abs(offered_price - expected_price) / expected_price if expected_price > 0 else 0
                
                if price_difference > 0.1:  # Significant arbitrage opportunity
                    return self.generate_message(
                        message.sender,
                        MessageType.ACCEPT,
                        {'trade_id': content.get('trade_id'), 'quick_execution': True}
                    )
                else:
                    # Counter with a more favorable price
                    counter_price = expected_price * 0.95 if offered_price > expected_price else expected_price * 1.05
                    return self.generate_message(
                        message.sender,
                        MessageType.COUNTER_OFFER,
                        {
                            'trade_id': content.get('trade_id'),
                            'counter_price': counter_price,
                            'quantity': content.get('quantity', 0)
                        }
                    )
        
        return None
    
    def calculate_portfolio_performance(self, observation: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate detailed portfolio performance metrics"""
        
        current_value = self.calculate_portfolio_value(dict(zip(ResourceType, observation['market_prices'])))
        
        # Calculate returns
        if self.pnl_history:
            total_return = (current_value - self.initial_cash) / self.initial_cash
            
            # Sharpe ratio (simplified)
            if len(self.pnl_history) > 5:
                returns = np.diff(self.pnl_history)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            peak = max(self.pnl_history)
            current_drawdown = (peak - current_value) / peak if peak > 0 else 0
            
        else:
            total_return = 0
            sharpe_ratio = 0
            current_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'current_drawdown': current_drawdown,
            'portfolio_value': current_value,
            'cash_ratio': self.cash / current_value if current_value > 0 else 1,
            'active_positions': len([pos for pos in self.positions.values() if pos > 0]),
            'trade_count': len(self.trade_history),
            'win_rate': len([t for t in self.trade_history if t.get('profit', 0) > 0]) / len(self.trade_history) if self.trade_history else 0
        }
    
    def get_market_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive market analysis report"""
        
        return {
            'volatility_estimates': dict(self.volatility_estimates),
            'momentum_indicators': dict(self.momentum_indicators),
            'support_resistance_levels': {rt.value: levels for rt, levels in self.support_resistance_levels.items()},
            'arbitrage_opportunities_found': len([op for op in self._find_arbitrage_opportunities({
                'market_prices': [100] * len(ResourceType),  # Dummy prices
                'portfolio': [50] * len(ResourceType)
            })]),
            'manipulation_risk_score': self.manipulation_risk_score,
            'information_network_size': len(self.information_sources),
            'recent_trade_frequency': self.recent_trades_count
        }