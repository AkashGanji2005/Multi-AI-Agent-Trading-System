# AI Negotiator: Multi-Agent Reinforcement Learning Marketplace

A sophisticated system of AI agents that autonomously negotiate, form alliances, and make decisions in a dynamic, decentralized marketplace environment using deep reinforcement learning, game theory, and communication protocols.

## 🎯 Features

- **Multi-Agent Environment**: 5 distinct agent types with different objectives
- **Dynamic Marketplace**: Simulates trading of energy, goods, or data
- **Advanced Communication**: Message passing and shared environment signals
- **Game Theory Integration**: Nash equilibrium and coalition formation
- **Reinforcement Learning**: MARL algorithms with RLLib and Stable-Baselines3
- **Real-time Visualization**: Interactive dashboard for monitoring negotiations

## 🤖 Agent Types

1. **Buyer**: Seeks to minimize costs while securing necessary resources
2. **Seller**: Maximizes profit through strategic pricing and negotiations
3. **Regulator**: Ensures fair trading practices and market stability
4. **Mediator**: Facilitates negotiations and resolves disputes
5. **Speculator**: Exploits market inefficiencies for profit

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-negotiator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Run a quick simulation test
python run_simulation.py

# 2. Launch the interactive dashboard
streamlit run dashboard.py

# 3. Run a full simulation with specific scenario
python main.py simulate --scenario energy_trading --max-steps 1500

# 4. Train agents using MARL
python main.py train --scenario balanced --iterations 1000 --workers 4

# 5. Evaluate trained model
python main.py evaluate --checkpoint checkpoints/checkpoint_000500 --eval-episodes 20

# 6. List available scenarios
python main.py scenarios
```

### Training Examples

```bash
# Train with PPO on energy trading scenario
python train.py --scenario energy_trading --algorithm PPO --iterations 1500

# Train with SAC on data marketplace
python train.py --scenario data_marketplace --algorithm SAC --iterations 1200

# Train with Weights & Biases logging
python train.py --scenario financial_market --wandb-project my-ai-negotiator
```

## 📊 Architecture

The system is built on:
- **PettingZoo**: Multi-agent environment framework
- **Ray RLLib**: Distributed reinforcement learning
- **PyTorch**: Deep learning backend
- **Game Theory**: Strategic decision making
- **ZMQ**: Inter-agent communication

## 🎮 Environment

The marketplace environment supports:
- Continuous and discrete action spaces
- Dynamic pricing mechanisms
- Resource scarcity simulation
- Communication channels
- Alliance formation
- Reputation systems

## 📈 Training

Supports multiple MARL algorithms:
- Multi-Agent PPO (MAPPO)
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- Independent Q-Learning
- Centralized Training with Decentralized Execution (CTDE)

## 🔧 Configuration

### Available Scenarios

- **Balanced**: Equal representation of all agent types (default)
- **Energy Trading**: Specialized energy marketplace with producers and consumers
- **Data Marketplace**: Privacy-focused data trading with quality verification
- **Financial Market**: High-frequency trading with sophisticated agents
- **Commodity Exchange**: Agricultural and raw materials with seasonal patterns
- **Multi-Market**: Complex ecosystem with interconnected markets

### Customization

Edit `config/scenarios.yaml` to customize:
- Market parameters (volatility, liquidity, etc.)
- Agent populations and distributions
- Resource types and initial allocations
- Communication protocols
- Reward structures and penalties
- Training hyperparameters

### Environment Features

- **Multi-Agent Support**: Up to 40+ agents with different objectives
- **Communication**: Message passing, negotiations, auctions
- **Alliances**: Dynamic coalition formation
- **Regulation**: Market oversight and violation penalties  
- **Game Theory**: Nash equilibrium and strategic interactions
- **Real-time Monitoring**: Live dashboard with network visualization

## 📈 Training Algorithms

Supported MARL algorithms:
- **PPO**: Proximal Policy Optimization (recommended)
- **DQN**: Deep Q-Network for discrete actions
- **SAC**: Soft Actor-Critic for continuous control
- **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient

## 🎯 Key Features

### Advanced Agent Behaviors
- **Strategic Pricing**: Dynamic pricing based on market conditions
- **Risk Management**: Portfolio optimization and position sizing
- **Reputation Systems**: Trust-based relationship building
- **Market Analysis**: Technical indicators and trend following
- **Regulatory Compliance**: Violation detection and penalties

### Communication Protocols
- **Direct Messaging**: Agent-to-agent communication
- **Broadcasting**: Public announcements
- **Auctions**: Competitive bidding mechanisms
- **Negotiations**: Structured multi-round bargaining
- **Gossip Protocol**: Information spreading

### Monitoring & Visualization
- **Real-time Dashboard**: Live market monitoring
- **Network Graphs**: Agent relationship visualization
- **Performance Metrics**: Trading success, cooperation scores
- **Market Analytics**: Price trends, volatility, efficiency

## 🗄️ Big Data Integration

### Supported Frameworks
- **Apache Spark**: Large-scale data processing and analytics
- **Apache Kafka**: Real-time streaming and event processing
- **Redis**: High-performance caching and real-time data
- **Elasticsearch**: Advanced search and log analytics
- **MongoDB**: Document-based storage for complex data
- **Cassandra**: Time-series data and high-availability storage
- **Dask**: Distributed computing for Python

### Big Data Use Cases
- **Real-time Market Monitoring**: Stream processing of trades and prices
- **Historical Analytics**: Large-scale pattern analysis across millions of trades
- **Anomaly Detection**: ML-based detection of market manipulation
- **Agent Behavior Clustering**: Unsupervised learning on agent strategies
- **Predictive Analytics**: Market forecasting using time-series analysis
- **Data Lake Integration**: Export to Parquet, Delta Lake formats

### Quick Start with Big Data

```bash
# Install big data dependencies
pip install pyspark kafka-python redis elasticsearch pymongo

# Run big data demonstration
python examples/big_data_example.py

# Start with specific frameworks
python -c "from src.data.spark_processor import create_sample_data_pipeline; create_sample_data_pipeline()"
```