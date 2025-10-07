"""
Multi-Agent Marketplace Environment using PettingZoo
Supports dynamic trading, negotiation, and alliance formation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from typing import Dict, List, Optional, Tuple, Any, Union
import random
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

class ResourceType(Enum):
    ENERGY = "energy"
    DATA = "data"
    GOODS = "goods"
    SERVICES = "services"

class AgentType(Enum):
    BUYER = "buyer"
    SELLER = "seller"
    REGULATOR = "regulator"
    MEDIATOR = "mediator"
    SPECULATOR = "speculator"

class MessageType(Enum):
    OFFER = "offer"
    COUNTER_OFFER = "counter_offer"
    ACCEPT = "accept"
    REJECT = "reject"
    ALLIANCE_PROPOSAL = "alliance_proposal"
    ALLIANCE_ACCEPT = "alliance_accept"
    ALLIANCE_REJECT = "alliance_reject"
    INFORMATION_REQUEST = "info_request"
    INFORMATION_SHARE = "info_share"
    DISPUTE = "dispute"
    MEDIATION_REQUEST = "mediation_request"

@dataclass
class Resource:
    type: ResourceType
    quantity: float
    quality: float  # 0-1 scale
    price: float
    owner: Optional[str] = None
    expiry_time: Optional[int] = None

@dataclass
class Trade:
    id: str
    buyer: str
    seller: str
    resource: Resource
    agreed_price: float
    timestamp: int
    status: str = "pending"  # pending, completed, failed, disputed
    mediator: Optional[str] = None

@dataclass
class Message:
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: int
    response_to: Optional[str] = None

@dataclass
class Alliance:
    id: str
    members: List[str]
    purpose: str
    profit_sharing: Dict[str, float]
    duration: int
    created_at: int

class MarketplaceEnv(ParallelEnv):
    """
    Multi-agent marketplace environment where agents negotiate, trade, and form alliances
    """
    
    metadata = {"render_modes": ["human"], "name": "marketplace_v1"}
    
    def __init__(self, 
                 num_agents: int = 10,
                 max_steps: int = 1000,
                 resource_types: List[ResourceType] = None,
                 initial_resources: Dict[str, int] = None,
                 communication_enabled: bool = True,
                 alliance_enabled: bool = True,
                 regulation_enabled: bool = True):
        
        super().__init__()
        
        # Environment parameters
        self._num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.resource_types = resource_types or list(ResourceType)
        self.initial_resources = initial_resources or {rt.value: 100 for rt in self.resource_types}
        self.communication_enabled = communication_enabled
        self.alliance_enabled = alliance_enabled
        self.regulation_enabled = regulation_enabled
        
        # Market state
        self.resources: Dict[str, List[Resource]] = {}
        self.trades: Dict[str, Trade] = {}
        self.messages: List[Message] = []
        self.alliances: Dict[str, Alliance] = {}
        self.market_prices: Dict[ResourceType, float] = {}
        self.reputation_scores: Dict[str, float] = {}
        self.violation_counts: Dict[str, int] = {}
        
        # Agent configuration
        self.agent_types: Dict[str, AgentType] = {}
        self.agent_portfolios: Dict[str, Dict[ResourceType, float]] = {}
        self.agent_cash: Dict[str, float] = {}
        self.agent_objectives: Dict[str, Dict[str, float]] = {}
        
        # Communication network
        self.communication_network = nx.Graph()
        
        # Initialize agents
        self._initialize_agents()
        
        # Define action and observation spaces
        self._setup_spaces()
        
    def _initialize_agents(self):
        """Initialize agents with types, resources, and objectives"""
        
        # Distribute agent types
        type_distribution = {
            AgentType.BUYER: 0.3,
            AgentType.SELLER: 0.3,
            AgentType.REGULATOR: 0.1,
            AgentType.MEDIATOR: 0.1,
            AgentType.SPECULATOR: 0.2
        }
        
        agent_types = []
        for agent_type, ratio in type_distribution.items():
            count = max(1, int(self.num_agents * ratio))
            agent_types.extend([agent_type] * count)
        
        # Ensure we have exactly num_agents
        while len(agent_types) < self.num_agents:
            agent_types.append(random.choice(list(AgentType)))
        agent_types = agent_types[:self.num_agents]
        random.shuffle(agent_types)
        
        # Initialize each agent
        for i, agent_type in enumerate(agent_types):
            agent_id = f"agent_{i}"
            self.possible_agents.append(agent_id)
            self.agent_types[agent_id] = agent_type
            
            # Initialize portfolio and cash based on agent type
            if agent_type == AgentType.BUYER:
                self.agent_cash[agent_id] = random.uniform(1000, 5000)
                self.agent_portfolios[agent_id] = {rt: random.uniform(0, 10) for rt in self.resource_types}
                self.agent_objectives[agent_id] = {"minimize_cost": 0.8, "maximize_quality": 0.2}
                
            elif agent_type == AgentType.SELLER:
                self.agent_cash[agent_id] = random.uniform(500, 2000)
                self.agent_portfolios[agent_id] = {rt: random.uniform(50, 200) for rt in self.resource_types}
                self.agent_objectives[agent_id] = {"maximize_profit": 0.9, "maintain_reputation": 0.1}
                
            elif agent_type == AgentType.REGULATOR:
                self.agent_cash[agent_id] = random.uniform(10000, 50000)
                self.agent_portfolios[agent_id] = {rt: 0 for rt in self.resource_types}
                self.agent_objectives[agent_id] = {"ensure_fairness": 0.7, "prevent_monopoly": 0.3}
                
            elif agent_type == AgentType.MEDIATOR:
                self.agent_cash[agent_id] = random.uniform(2000, 10000)
                self.agent_portfolios[agent_id] = {rt: random.uniform(10, 50) for rt in self.resource_types}
                self.agent_objectives[agent_id] = {"resolve_disputes": 0.6, "earn_fees": 0.4}
                
            elif agent_type == AgentType.SPECULATOR:
                self.agent_cash[agent_id] = random.uniform(5000, 20000)
                self.agent_portfolios[agent_id] = {rt: random.uniform(20, 100) for rt in self.resource_types}
                self.agent_objectives[agent_id] = {"maximize_profit": 1.0}
            
            # Initialize reputation and violation tracking
            self.reputation_scores[agent_id] = 0.5  # Start neutral
            self.violation_counts[agent_id] = 0
            
            # Add to communication network
            self.communication_network.add_node(agent_id)
        
        # Create initial market prices
        for resource_type in self.resource_types:
            self.market_prices[resource_type] = random.uniform(10, 100)
            
        # Create communication connections (small-world network)
        for agent_id in self.possible_agents:
            # Connect to a few random agents
            connections = random.sample([a for a in self.possible_agents if a != agent_id], 
                                      min(3, len(self.possible_agents) - 1))
            for other_agent in connections:
                self.communication_network.add_edge(agent_id, other_agent)
    
    def _setup_spaces(self):
        """Define action and observation spaces for each agent"""
        
        # Action space: [trade_action, communication_action, alliance_action]
        # Trade action: [resource_type, quantity, price, target_agent]
        # Communication action: [message_type, target_agent, content_params...]
        # Alliance action: [action_type, target_agents...]
        
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for agent_id in self.possible_agents:
            # Action space
            action_space = spaces.Dict({
                # Trading actions
                "trade_resource_type": spaces.Discrete(len(self.resource_types)),
                "trade_quantity": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
                "trade_price": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
                "trade_target": spaces.Discrete(self.num_agents),
                "trade_action_type": spaces.Discrete(4),  # 0: buy, 1: sell, 2: no_action, 3: cancel
                
                # Communication actions
                "comm_message_type": spaces.Discrete(len(MessageType)),
                "comm_target": spaces.Discrete(self.num_agents),
                "comm_enabled": spaces.Discrete(2),  # 0: no message, 1: send message
                
                # Alliance actions
                "alliance_action": spaces.Discrete(4),  # 0: none, 1: propose, 2: accept, 3: leave
                "alliance_target": spaces.Discrete(self.num_agents),
            })
            
            # Observation space
            obs_space = spaces.Dict({
                # Agent's own state
                "cash": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "portfolio": spaces.Box(low=0, high=np.inf, shape=(len(self.resource_types),), dtype=np.float32),
                "reputation": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "agent_type": spaces.Discrete(len(AgentType)),
                
                # Market state
                "market_prices": spaces.Box(low=0, high=np.inf, shape=(len(self.resource_types),), dtype=np.float32),
                "time_step": spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.float32),
                
                # Other agents (observable info)
                "other_agents_reputation": spaces.Box(low=0, high=1, shape=(self.num_agents-1,), dtype=np.float32),
                "other_agents_types": spaces.Box(low=0, high=len(AgentType)-1, shape=(self.num_agents-1,), dtype=np.int32),
                
                # Recent messages (last 5)
                "recent_messages": spaces.Box(low=0, high=1, shape=(5, 10), dtype=np.float32),  # Encoded messages
                
                # Active trades
                "active_trades": spaces.Box(low=0, high=1, shape=(10, 8), dtype=np.float32),  # Encoded trades
                
                # Alliance memberships
                "alliance_memberships": spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
            })
            
            self.action_spaces[agent_id] = action_space
            self.observation_spaces[agent_id] = obs_space
        
        self.agents = self.possible_agents[:]
    
    @property
    def num_agents(self) -> int:
        return self._num_agents
    
    @property
    def possible_agents(self) -> List[str]:
        if not hasattr(self, '_possible_agents'):
            self._possible_agents = []
        return self._possible_agents
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_step = 0
        self.agents = self.possible_agents[:]
        
        # Reset market state
        self.resources.clear()
        self.trades.clear()
        self.messages.clear()
        self.alliances.clear()
        
        # Reset market prices with some volatility
        for resource_type in self.resource_types:
            self.market_prices[resource_type] = random.uniform(10, 100)
        
        # Reset agent states
        for agent_id in self.possible_agents:
            self.reputation_scores[agent_id] = 0.5
            self.violation_counts[agent_id] = 0
        
        # Re-initialize agent portfolios and cash with some randomness
        self._initialize_agents()
        
        # Get initial observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)
        
        infos = {agent_id: {} for agent_id in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, Any]):
        """Execute one environment step"""
        self.current_step += 1
        
        # Process actions for each agent
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Phase 1: Process trade actions
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue
                
            reward = 0.0
            
            # Process trading action
            reward += self._process_trade_action(agent_id, action)
            
            # Process communication action
            if self.communication_enabled:
                reward += self._process_communication_action(agent_id, action)
            
            # Process alliance action
            if self.alliance_enabled:
                reward += self._process_alliance_action(agent_id, action)
            
            rewards[agent_id] = reward
            terminations[agent_id] = False
            truncations[agent_id] = self.current_step >= self.max_steps
            infos[agent_id] = self._get_agent_info(agent_id)
        
        # Phase 2: Update market dynamics
        self._update_market_prices()
        self._process_pending_trades()
        self._update_reputation_scores()
        self._cleanup_expired_resources()
        
        # Phase 3: Regulatory actions
        if self.regulation_enabled:
            self._apply_regulatory_actions()
        
        # Get new observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)
        
        return observations, rewards, terminations, truncations, infos
    
    def _process_trade_action(self, agent_id: str, action: Dict[str, Any]) -> float:
        """Process trading actions and return reward"""
        reward = 0.0
        
        trade_action_type = action.get("trade_action_type", 2)  # Default: no action
        
        if trade_action_type == 2:  # No action
            return 0.0
        
        resource_type_idx = action.get("trade_resource_type", 0)
        resource_type = self.resource_types[resource_type_idx]
        quantity = float(action.get("trade_quantity", [0])[0])
        price = float(action.get("trade_price", [0])[0])
        target_idx = action.get("trade_target", 0)
        target_agent = self.possible_agents[target_idx % len(self.possible_agents)]
        
        if target_agent == agent_id:  # Can't trade with self
            return -0.1
        
        if trade_action_type == 0:  # Buy
            reward += self._process_buy_action(agent_id, target_agent, resource_type, quantity, price)
        elif trade_action_type == 1:  # Sell
            reward += self._process_sell_action(agent_id, target_agent, resource_type, quantity, price)
        elif trade_action_type == 3:  # Cancel existing trades
            reward += self._cancel_agent_trades(agent_id)
        
        return reward
    
    def _process_buy_action(self, buyer_id: str, seller_id: str, resource_type: ResourceType, 
                          quantity: float, offered_price: float) -> float:
        """Process a buy action"""
        
        # Check if buyer has enough cash
        total_cost = quantity * offered_price
        if self.agent_cash[buyer_id] < total_cost:
            return -0.5  # Penalty for invalid action
        
        # Check if seller has the resource
        if self.agent_portfolios[seller_id][resource_type] < quantity:
            return -0.1  # Small penalty, might negotiate
        
        # Create trade proposal
        trade_id = f"trade_{len(self.trades)}_{self.current_step}"
        resource = Resource(
            type=resource_type,
            quantity=quantity,
            quality=random.uniform(0.5, 1.0),  # Random quality
            price=offered_price,
            owner=seller_id
        )
        
        trade = Trade(
            id=trade_id,
            buyer=buyer_id,
            seller=seller_id,
            resource=resource,
            agreed_price=offered_price,
            timestamp=self.current_step,
            status="pending"
        )
        
        self.trades[trade_id] = trade
        
        # Simple acceptance logic (can be made more sophisticated)
        market_price = self.market_prices[resource_type]
        acceptance_probability = min(0.9, offered_price / market_price)
        
        if random.random() < acceptance_probability:
            return self._execute_trade(trade_id)
        
        return 0.1  # Small reward for making a reasonable offer
    
    def _process_sell_action(self, seller_id: str, buyer_id: str, resource_type: ResourceType,
                           quantity: float, asking_price: float) -> float:
        """Process a sell action"""
        
        # Check if seller has the resource
        if self.agent_portfolios[seller_id][resource_type] < quantity:
            return -0.5  # Penalty for invalid action
        
        # Check if buyer might be interested (has cash)
        total_cost = quantity * asking_price
        if self.agent_cash[buyer_id] < total_cost:
            return -0.1  # Buyer can't afford, small penalty
        
        # Create trade proposal
        trade_id = f"trade_{len(self.trades)}_{self.current_step}"
        resource = Resource(
            type=resource_type,
            quantity=quantity,
            quality=random.uniform(0.5, 1.0),
            price=asking_price,
            owner=seller_id
        )
        
        trade = Trade(
            id=trade_id,
            buyer=buyer_id,
            seller=seller_id,
            resource=resource,
            agreed_price=asking_price,
            timestamp=self.current_step,
            status="pending"
        )
        
        self.trades[trade_id] = trade
        
        # Simple acceptance logic
        market_price = self.market_prices[resource_type]
        acceptance_probability = min(0.9, market_price / asking_price)
        
        if random.random() < acceptance_probability:
            return self._execute_trade(trade_id)
        
        return 0.1  # Small reward for making a reasonable offer
    
    def _execute_trade(self, trade_id: str) -> float:
        """Execute a trade and return reward"""
        trade = self.trades[trade_id]
        
        buyer_id = trade.buyer
        seller_id = trade.seller
        resource = trade.resource
        total_cost = resource.quantity * trade.agreed_price
        
        # Transfer resources and cash
        self.agent_cash[buyer_id] -= total_cost
        self.agent_cash[seller_id] += total_cost
        self.agent_portfolios[buyer_id][resource.type] += resource.quantity
        self.agent_portfolios[seller_id][resource.type] -= resource.quantity
        
        # Update trade status
        trade.status = "completed"
        
        # Calculate rewards based on market efficiency
        market_price = self.market_prices[resource.type]
        buyer_reward = max(0, (market_price - trade.agreed_price) * resource.quantity / 100)
        seller_reward = max(0, (trade.agreed_price - market_price) * resource.quantity / 100)
        
        # Update reputation scores
        self.reputation_scores[buyer_id] = min(1.0, self.reputation_scores[buyer_id] + 0.01)
        self.reputation_scores[seller_id] = min(1.0, self.reputation_scores[seller_id] + 0.01)
        
        return buyer_reward if trade.buyer else seller_reward
    
    def _process_communication_action(self, agent_id: str, action: Dict[str, Any]) -> float:
        """Process communication actions"""
        if not action.get("comm_enabled", 0):
            return 0.0
        
        message_type_idx = action.get("comm_message_type", 0)
        message_type = list(MessageType)[message_type_idx]
        target_idx = action.get("comm_target", 0)
        target_agent = self.possible_agents[target_idx % len(self.possible_agents)]
        
        if target_agent == agent_id:
            return -0.1  # Can't message self
        
        # Check if agents are connected in communication network
        if not self.communication_network.has_edge(agent_id, target_agent):
            return -0.05  # Penalty for trying to communicate with unconnected agent
        
        # Create message
        message = Message(
            sender=agent_id,
            receiver=target_agent,
            message_type=message_type,
            content={"timestamp": self.current_step},
            timestamp=self.current_step
        )
        
        self.messages.append(message)
        
        # Small reward for communication
        return 0.05
    
    def _process_alliance_action(self, agent_id: str, action: Dict[str, Any]) -> float:
        """Process alliance formation actions"""
        alliance_action = action.get("alliance_action", 0)
        
        if alliance_action == 0:  # No action
            return 0.0
        
        target_idx = action.get("alliance_target", 0)
        target_agent = self.possible_agents[target_idx % len(self.possible_agents)]
        
        if alliance_action == 1:  # Propose alliance
            return self._propose_alliance(agent_id, target_agent)
        elif alliance_action == 2:  # Accept alliance
            return self._accept_alliance(agent_id, target_agent)
        elif alliance_action == 3:  # Leave alliance
            return self._leave_alliance(agent_id)
        
        return 0.0
    
    def _propose_alliance(self, proposer: str, target: str) -> float:
        """Propose an alliance between agents"""
        if proposer == target:
            return -0.1
        
        # Simple alliance acceptance logic
        proposer_rep = self.reputation_scores[proposer]
        target_rep = self.reputation_scores[target]
        
        if proposer_rep > 0.6 and target_rep > 0.6:
            alliance_id = f"alliance_{len(self.alliances)}_{self.current_step}"
            alliance = Alliance(
                id=alliance_id,
                members=[proposer, target],
                purpose="mutual_benefit",
                profit_sharing={proposer: 0.5, target: 0.5},
                duration=100,  # 100 steps
                created_at=self.current_step
            )
            self.alliances[alliance_id] = alliance
            return 0.2  # Reward for successful alliance
        
        return 0.05  # Small reward for trying
    
    def _accept_alliance(self, agent_id: str, proposer: str) -> float:
        """Accept an alliance proposal"""
        # Find pending alliance proposal
        for alliance in self.alliances.values():
            if proposer in alliance.members and agent_id not in alliance.members:
                alliance.members.append(agent_id)
                alliance.profit_sharing[agent_id] = 1.0 / len(alliance.members)
                # Rebalance profit sharing
                for member in alliance.members:
                    alliance.profit_sharing[member] = 1.0 / len(alliance.members)
                return 0.15
        return 0.0
    
    def _leave_alliance(self, agent_id: str) -> float:
        """Leave current alliance"""
        for alliance_id, alliance in list(self.alliances.items()):
            if agent_id in alliance.members:
                alliance.members.remove(agent_id)
                del alliance.profit_sharing[agent_id]
                if len(alliance.members) < 2:
                    del self.alliances[alliance_id]
                else:
                    # Rebalance profit sharing
                    for member in alliance.members:
                        alliance.profit_sharing[member] = 1.0 / len(alliance.members)
                return -0.1  # Small penalty for leaving
        return 0.0
    
    def _cancel_agent_trades(self, agent_id: str) -> float:
        """Cancel all pending trades for an agent"""
        cancelled_count = 0
        for trade in self.trades.values():
            if (trade.buyer == agent_id or trade.seller == agent_id) and trade.status == "pending":
                trade.status = "cancelled"
                cancelled_count += 1
        
        return -0.05 * cancelled_count  # Small penalty for cancelling
    
    def _update_market_prices(self):
        """Update market prices based on supply and demand"""
        for resource_type in self.resource_types:
            # Calculate supply and demand
            total_supply = sum(self.agent_portfolios[agent][resource_type] 
                             for agent in self.possible_agents)
            
            # Count recent trades for this resource
            recent_trades = [t for t in self.trades.values() 
                           if t.resource.type == resource_type and 
                           t.timestamp > self.current_step - 10 and 
                           t.status == "completed"]
            
            if recent_trades:
                avg_price = np.mean([t.agreed_price for t in recent_trades])
                # Move market price towards average trade price
                self.market_prices[resource_type] = 0.9 * self.market_prices[resource_type] + 0.1 * avg_price
            
            # Add some random volatility
            volatility = random.uniform(-0.05, 0.05)
            self.market_prices[resource_type] *= (1 + volatility)
            self.market_prices[resource_type] = max(1.0, self.market_prices[resource_type])  # Minimum price
    
    def _process_pending_trades(self):
        """Process any pending trades that might be automatically executed"""
        for trade in self.trades.values():
            if trade.status == "pending" and self.current_step - trade.timestamp > 5:
                # Timeout old trades
                trade.status = "expired"
    
    def _update_reputation_scores(self):
        """Update reputation scores based on recent behavior"""
        for agent_id in self.possible_agents:
            # Decay reputation slightly over time
            self.reputation_scores[agent_id] *= 0.999
            
            # Penalize violations
            if self.violation_counts[agent_id] > 0:
                penalty = min(0.1, self.violation_counts[agent_id] * 0.02)
                self.reputation_scores[agent_id] = max(0.0, self.reputation_scores[agent_id] - penalty)
    
    def _cleanup_expired_resources(self):
        """Remove expired resources and clean up old data"""
        # Clean up old messages (keep only last 100)
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]
        
        # Clean up old trades (keep only last 1000)
        if len(self.trades) > 1000:
            recent_trades = dict(list(self.trades.items())[-1000:])
            self.trades = recent_trades
    
    def _apply_regulatory_actions(self):
        """Apply regulatory oversight and penalties"""
        regulators = [agent_id for agent_id, agent_type in self.agent_types.items() 
                     if agent_type == AgentType.REGULATOR]
        
        if not regulators:
            return
        
        # Check for market manipulation
        for agent_id in self.possible_agents:
            if self.agent_types[agent_id] == AgentType.SPECULATOR:
                # Check if speculator is manipulating prices
                recent_trades = [t for t in self.trades.values() 
                               if (t.buyer == agent_id or t.seller == agent_id) and 
                               t.timestamp > self.current_step - 20]
                
                if len(recent_trades) > 10:  # Too many trades
                    self.violation_counts[agent_id] += 1
                    # Regulatory fine
                    fine = min(self.agent_cash[agent_id] * 0.1, 1000)
                    self.agent_cash[agent_id] -= fine
                    # Distribute fine to regulators
                    for regulator in regulators:
                        self.agent_cash[regulator] += fine / len(regulators)
    
    def _get_observation(self, agent_id: str) -> Dict[str, np.ndarray]:
        """Get observation for a specific agent"""
        
        # Agent's own state
        cash = np.array([self.agent_cash[agent_id]], dtype=np.float32)
        portfolio = np.array([self.agent_portfolios[agent_id][rt] for rt in self.resource_types], dtype=np.float32)
        reputation = np.array([self.reputation_scores[agent_id]], dtype=np.float32)
        agent_type_idx = list(AgentType).index(self.agent_types[agent_id])
        agent_type = np.array([agent_type_idx], dtype=np.int32)
        
        # Market state
        market_prices = np.array([self.market_prices[rt] for rt in self.resource_types], dtype=np.float32)
        time_step = np.array([self.current_step / self.max_steps], dtype=np.float32)
        
        # Other agents
        other_agents = [a for a in self.possible_agents if a != agent_id]
        other_reps = np.array([self.reputation_scores[a] for a in other_agents], dtype=np.float32)
        other_types = np.array([list(AgentType).index(self.agent_types[a]) for a in other_agents], dtype=np.int32)
        
        # Recent messages (simplified encoding)
        recent_msgs = [m for m in self.messages[-5:] if m.receiver == agent_id or m.sender == agent_id]
        msg_encoding = np.zeros((5, 10), dtype=np.float32)
        for i, msg in enumerate(recent_msgs[:5]):
            msg_encoding[i, 0] = list(MessageType).index(msg.message_type) / len(MessageType)
            msg_encoding[i, 1] = self.possible_agents.index(msg.sender) / len(self.possible_agents)
            msg_encoding[i, 2] = msg.timestamp / self.max_steps
        
        # Active trades (simplified encoding)
        agent_trades = [t for t in self.trades.values() 
                       if (t.buyer == agent_id or t.seller == agent_id) and t.status == "pending"]
        trade_encoding = np.zeros((10, 8), dtype=np.float32)
        for i, trade in enumerate(agent_trades[:10]):
            trade_encoding[i, 0] = list(ResourceType).index(trade.resource.type) / len(ResourceType)
            trade_encoding[i, 1] = trade.resource.quantity / 1000  # Normalize
            trade_encoding[i, 2] = trade.agreed_price / 1000  # Normalize
            trade_encoding[i, 3] = 1.0 if trade.buyer == agent_id else 0.0
            trade_encoding[i, 4] = trade.resource.quality
            trade_encoding[i, 5] = trade.timestamp / self.max_steps
        
        # Alliance memberships
        alliance_members = np.zeros(self.num_agents, dtype=np.float32)
        for alliance in self.alliances.values():
            if agent_id in alliance.members:
                for member in alliance.members:
                    if member != agent_id:
                        idx = self.possible_agents.index(member)
                        alliance_members[idx] = 1.0
        
        return {
            "cash": cash,
            "portfolio": portfolio,
            "reputation": reputation,
            "agent_type": agent_type,
            "market_prices": market_prices,
            "time_step": time_step,
            "other_agents_reputation": other_reps,
            "other_agents_types": other_types,
            "recent_messages": msg_encoding,
            "active_trades": trade_encoding,
            "alliance_memberships": alliance_members
        }
    
    def _get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get additional info for an agent"""
        return {
            "cash": self.agent_cash[agent_id],
            "portfolio": dict(self.agent_portfolios[agent_id]),
            "reputation": self.reputation_scores[agent_id],
            "agent_type": self.agent_types[agent_id].value,
            "violations": self.violation_counts[agent_id],
            "active_trades": len([t for t in self.trades.values() 
                                if (t.buyer == agent_id or t.seller == agent_id) and t.status == "pending"]),
            "alliances": [a.id for a in self.alliances.values() if agent_id in a.members]
        }
    
    def render(self):
        """Render the environment (basic text output)"""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Active Agents: {len(self.agents)}")
        print(f"Market Prices: {dict(self.market_prices)}")
        print(f"Active Trades: {len([t for t in self.trades.values() if t.status == 'pending'])}")
        print(f"Alliances: {len(self.alliances)}")
        print("-" * 50)

def env():
    """Environment factory function"""
    return MarketplaceEnv()

# Register environment
from pettingzoo.utils import wrappers

def make_env():
    """Create wrapped environment"""
    env = MarketplaceEnv()
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env