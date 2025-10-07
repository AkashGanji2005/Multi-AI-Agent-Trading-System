"""
Agent Factory for creating and managing different types of marketplace agents
"""

from typing import Dict, Any, List, Optional, Type
import random

from .base_agent import BaseAgent
from .buyer_agent import BuyerAgent
from .seller_agent import SellerAgent
from .regulator_agent import RegulatorAgent
from .mediator_agent import MediatorAgent
from .speculator_agent import SpeculatorAgent
from ..environment.marketplace import AgentType, ResourceType

class AgentFactory:
    """Factory class for creating marketplace agents"""
    
    AGENT_CLASSES = {
        AgentType.BUYER: BuyerAgent,
        AgentType.SELLER: SellerAgent,
        AgentType.REGULATOR: RegulatorAgent,
        AgentType.MEDIATOR: MediatorAgent,
        AgentType.SPECULATOR: SpeculatorAgent
    }
    
    @classmethod
    def create_agent(cls, 
                     agent_id: str, 
                     agent_type: AgentType,
                     **kwargs) -> BaseAgent:
        """Create an agent of the specified type"""
        
        if agent_type not in cls.AGENT_CLASSES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls.AGENT_CLASSES[agent_type]
        
        # Set default parameters based on agent type
        default_params = cls._get_default_parameters(agent_type)
        default_params.update(kwargs)
        
        # Filter parameters to only include those accepted by BaseAgent
        valid_params = {
            'initial_cash': default_params.get('initial_cash', 1000.0),
            'initial_portfolio': default_params.get('initial_portfolio', {}),
            'objectives': default_params.get('objectives', {}),
            'learning_rate': default_params.get('learning_rate', 3e-4),
            'memory_size': default_params.get('memory_size', 10000)
        }
        
        return agent_class(agent_id, **valid_params)
    
    @classmethod
    def create_agent_population(cls, 
                               population_config: Dict[AgentType, int],
                               base_config: Dict[str, Any] = None) -> List[BaseAgent]:
        """Create a population of agents based on configuration"""
        
        base_config = base_config or {}
        agents = []
        agent_counter = 0
        
        for agent_type, count in population_config.items():
            for i in range(count):
                agent_id = f"{agent_type.value}_{i}"
                
                # Add some randomization to agent parameters
                agent_config = base_config.copy()
                agent_config.update(cls._get_randomized_parameters(agent_type))
                
                agent = cls.create_agent(agent_id, agent_type, **agent_config)
                agents.append(agent)
                agent_counter += 1
        
        return agents
    
    @classmethod
    def _get_default_parameters(cls, agent_type: AgentType) -> Dict[str, Any]:
        """Get default parameters for each agent type"""
        
        base_params = {
            'learning_rate': 3e-4,
            'memory_size': 10000
        }
        
        if agent_type == AgentType.BUYER:
            return {
                **base_params,
                'initial_cash': random.uniform(1000, 5000),
                'initial_portfolio': {rt: random.uniform(5, 25) for rt in ResourceType},
                'objectives': {'minimize_cost': 0.8, 'maximize_quality': 0.2}
            }
        
        elif agent_type == AgentType.SELLER:
            return {
                **base_params,
                'initial_cash': random.uniform(500, 3000),
                'initial_portfolio': {rt: random.uniform(50, 200) for rt in ResourceType},
                'objectives': {'maximize_profit': 0.9, 'maintain_reputation': 0.1}
            }
        
        elif agent_type == AgentType.REGULATOR:
            return {
                **base_params,
                'initial_cash': random.uniform(10000, 50000),
                'initial_portfolio': {rt: 0 for rt in ResourceType},
                'objectives': {'ensure_fairness': 0.7, 'prevent_monopoly': 0.3},
                'learning_rate': 1e-4  # Slower learning for regulator
            }
        
        elif agent_type == AgentType.MEDIATOR:
            return {
                **base_params,
                'initial_cash': random.uniform(2000, 10000),
                'initial_portfolio': {rt: random.uniform(10, 50) for rt in ResourceType},
                'objectives': {'resolve_disputes': 0.6, 'earn_fees': 0.4}
            }
        
        elif agent_type == AgentType.SPECULATOR:
            return {
                **base_params,
                'initial_cash': random.uniform(5000, 25000),
                'initial_portfolio': {rt: random.uniform(20, 100) for rt in ResourceType},
                'objectives': {'maximize_profit': 1.0},
                'learning_rate': 5e-4  # Faster learning for speculator
            }
        
        return base_params
    
    @classmethod
    def _get_randomized_parameters(cls, agent_type: AgentType) -> Dict[str, Any]:
        """Get randomized parameters to create diverse agents"""
        
        params = {}
        
        # Randomize personality traits
        params['risk_tolerance'] = random.uniform(0.1, 0.9)
        params['negotiation_patience'] = random.uniform(0.3, 0.9)
        params['cooperation_tendency'] = random.uniform(0.2, 0.8)
        params['price_aggressiveness'] = random.uniform(0.2, 0.8)
        
        # Agent-specific randomization
        if agent_type == AgentType.BUYER:
            params['quality_threshold'] = random.uniform(0.4, 0.8)
            params['urgency_factor'] = random.uniform(0.5, 1.0)
            params['max_price_premium'] = random.uniform(0.1, 0.3)
            
        elif agent_type == AgentType.SELLER:
            params['profit_margin_target'] = random.uniform(0.2, 0.5)
            params['inventory_turnover_target'] = random.uniform(0.6, 0.9)
            params['price_elasticity'] = random.uniform(0.3, 0.7)
            
        elif agent_type == AgentType.REGULATOR:
            params['fairness_threshold'] = random.uniform(0.6, 0.8)
            params['price_manipulation_threshold'] = random.uniform(0.2, 0.4)
            params['monopoly_threshold'] = random.uniform(0.5, 0.7)
            
        elif agent_type == AgentType.MEDIATOR:
            params['base_mediation_fee'] = random.uniform(30, 80)
            params['success_bonus_rate'] = random.uniform(0.05, 0.15)
            params['reputation_weight'] = random.uniform(0.6, 0.9)
            
        elif agent_type == AgentType.SPECULATOR:
            params['profit_target'] = random.uniform(0.3, 0.8)
            params['max_drawdown'] = random.uniform(0.2, 0.4)
            params['leverage_limit'] = random.uniform(2.0, 4.0)
        
        return params
    
    @classmethod
    def create_balanced_population(cls, 
                                  total_agents: int,
                                  custom_distribution: Dict[AgentType, float] = None) -> List[BaseAgent]:
        """Create a balanced population with specified or default distribution"""
        
        # Default distribution
        default_distribution = {
            AgentType.BUYER: 0.3,
            AgentType.SELLER: 0.3,
            AgentType.REGULATOR: 0.1,
            AgentType.MEDIATOR: 0.1,
            AgentType.SPECULATOR: 0.2
        }
        
        distribution = custom_distribution or default_distribution
        
        # Ensure distribution sums to 1.0
        total_ratio = sum(distribution.values())
        if abs(total_ratio - 1.0) > 0.01:
            distribution = {k: v / total_ratio for k, v in distribution.items()}
        
        # Calculate agent counts
        population_config = {}
        remaining_agents = total_agents
        
        for agent_type_str, ratio in distribution.items():
            # Convert string to AgentType enum
            if isinstance(agent_type_str, str):
                agent_type = AgentType(agent_type_str)
            else:
                agent_type = agent_type_str
            
            count = int(total_agents * ratio)
            population_config[agent_type] = count
            remaining_agents -= count
        
        # Distribute remaining agents
        while remaining_agents > 0:
            agent_type = random.choice(list(AgentType))
            population_config[agent_type] = population_config.get(agent_type, 0) + 1
            remaining_agents -= 1
        
        return cls.create_agent_population(population_config)
    
    @classmethod
    def create_scenario_agents(cls, scenario_name: str, **kwargs) -> List[BaseAgent]:
        """Create agents for specific scenarios"""
        
        if scenario_name == "energy_trading":
            return cls._create_energy_trading_agents(**kwargs)
        elif scenario_name == "data_marketplace":
            return cls._create_data_marketplace_agents(**kwargs)
        elif scenario_name == "commodity_exchange":
            return cls._create_commodity_exchange_agents(**kwargs)
        elif scenario_name == "financial_market":
            return cls._create_financial_market_agents(**kwargs)
        else:
            # Default balanced scenario
            total_agents = kwargs.get('total_agents', 15)
            return cls.create_balanced_population(total_agents)
    
    @classmethod
    def _create_energy_trading_agents(cls, **kwargs) -> List[BaseAgent]:
        """Create agents for energy trading scenario"""
        
        total_agents = kwargs.get('total_agents', 20)
        
        # Energy trading specific distribution
        distribution = {
            AgentType.BUYER: 0.4,    # Energy consumers
            AgentType.SELLER: 0.35,  # Energy producers
            AgentType.REGULATOR: 0.1, # Grid operator
            AgentType.MEDIATOR: 0.05, # Energy broker
            AgentType.SPECULATOR: 0.1 # Energy trader
        }
        
        agents = cls.create_balanced_population(total_agents, distribution)
        
        # Customize for energy trading
        for agent in agents:
            if agent.agent_type == AgentType.BUYER:
                # Energy consumers need consistent supply
                agent.urgency_factor = random.uniform(0.8, 1.0)
                agent.quality_threshold = random.uniform(0.7, 0.9)
                
            elif agent.agent_type == AgentType.SELLER:
                # Energy producers have capacity constraints
                agent.inventory_turnover_target = random.uniform(0.8, 1.0)
                
            elif agent.agent_type == AgentType.SPECULATOR:
                # Energy markets are volatile
                agent.volatility_trading_weight = 0.9
        
        return agents
    
    @classmethod
    def _create_data_marketplace_agents(cls, **kwargs) -> List[BaseAgent]:
        """Create agents for data marketplace scenario"""
        
        total_agents = kwargs.get('total_agents', 15)
        
        distribution = {
            AgentType.BUYER: 0.35,   # Data consumers
            AgentType.SELLER: 0.25,  # Data providers
            AgentType.REGULATOR: 0.15, # Privacy regulator
            AgentType.MEDIATOR: 0.15,  # Data broker
            AgentType.SPECULATOR: 0.1  # Data trader
        }
        
        agents = cls.create_balanced_population(total_agents, distribution)
        
        # Customize for data trading
        for agent in agents:
            if agent.agent_type == AgentType.BUYER:
                # Data buyers value quality highly
                agent.objectives = {'minimize_cost': 0.4, 'maximize_quality': 0.6}
                
            elif agent.agent_type == AgentType.REGULATOR:
                # Strong privacy enforcement
                agent.fairness_threshold = 0.8
                
            elif agent.agent_type == AgentType.MEDIATOR:
                # Data brokers earn higher fees
                agent.base_mediation_fee = random.uniform(80, 150)
        
        return agents
    
    @classmethod
    def _create_commodity_exchange_agents(cls, **kwargs) -> List[BaseAgent]:
        """Create agents for commodity exchange scenario"""
        
        total_agents = kwargs.get('total_agents', 25)
        
        distribution = {
            AgentType.BUYER: 0.3,
            AgentType.SELLER: 0.3,
            AgentType.REGULATOR: 0.05,
            AgentType.MEDIATOR: 0.1,
            AgentType.SPECULATOR: 0.25  # High speculation in commodities
        }
        
        agents = cls.create_balanced_population(total_agents, distribution)
        
        # Customize for commodity trading
        for agent in agents:
            if agent.agent_type == AgentType.SPECULATOR:
                # Commodity speculators use more leverage
                agent.leverage_limit = random.uniform(3.0, 6.0)
                agent.profit_target = random.uniform(0.4, 1.0)
        
        return agents
    
    @classmethod
    def _create_financial_market_agents(cls, **kwargs) -> List[BaseAgent]:
        """Create agents for financial market scenario"""
        
        total_agents = kwargs.get('total_agents', 30)
        
        distribution = {
            AgentType.BUYER: 0.25,   # Investors
            AgentType.SELLER: 0.25,  # Issuers
            AgentType.REGULATOR: 0.1, # Financial regulator
            AgentType.MEDIATOR: 0.05, # Financial advisor
            AgentType.SPECULATOR: 0.35 # Traders and funds
        }
        
        agents = cls.create_balanced_population(total_agents, distribution)
        
        # Customize for financial markets
        for agent in agents:
            if agent.agent_type == AgentType.REGULATOR:
                # Strict financial regulation
                agent.price_manipulation_threshold = 0.15
                agent.investigation_cooldown = 5
                
            elif agent.agent_type == AgentType.SPECULATOR:
                # Sophisticated financial strategies
                agent.strategies['arbitrage'] = 0.9
                agent.strategies['momentum'] = 0.8
        
        return agents

def create_test_population() -> List[BaseAgent]:
    """Create a small test population for development/testing"""
    
    population_config = {
        AgentType.BUYER: 2,
        AgentType.SELLER: 2,
        AgentType.REGULATOR: 1,
        AgentType.MEDIATOR: 1,
        AgentType.SPECULATOR: 2
    }
    
    return AgentFactory.create_agent_population(population_config)