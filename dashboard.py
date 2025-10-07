#!/usr/bin/env python3
"""
AI Negotiator Dashboard
Streamlit-based visualization and monitoring dashboard for the marketplace
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
import os
import sys
import yaml
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.marketplace import MarketplaceEnv, AgentType, ResourceType
from src.agents.agent_factory import AgentFactory
from src.communication.protocol import CommunicationManager

# Page configuration
st.set_page_config(
    page_title="AI Negotiator Dashboard",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .agent-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'step_data' not in st.session_state:
    st.session_state.step_data = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

@st.cache_data
def load_scenarios():
    """Load available scenarios"""
    config_path = "config/scenarios.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def create_agent_network_graph(agents, alliances):
    """Create network graph of agent relationships"""
    
    G = nx.Graph()
    
    # Add nodes (agents)
    for agent in agents:
        G.add_node(agent.agent_id, 
                  type=agent.agent_type.value,
                  reputation=agent.reputation,
                  cash=agent.cash)
    
    # Add edges (alliances and trust relationships)
    for alliance in alliances.values():
        members = alliance.members
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                G.add_edge(members[i], members[j], type='alliance')
    
    # Add trust relationships
    for agent in agents:
        for trusted_agent in agent.trusted_agents:
            if trusted_agent in [a.agent_id for a in agents]:
                G.add_edge(agent.agent_id, trusted_agent, type='trust')
    
    return G

def plot_network_graph(G):
    """Plot agent network using plotly"""
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edges
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(edge[2].get('type', 'unknown'))
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    color_map = {
        'buyer': '#1f77b4',
        'seller': '#ff7f0e',
        'regulator': '#2ca02c',
        'mediator': '#d62728',
        'speculator': '#9467bd'
    }
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        
        agent_type = node[1].get('type', 'unknown')
        reputation = node[1].get('reputation', 0.5)
        cash = node[1].get('cash', 0)
        
        node_text.append(f"{node[0]}<br>Type: {agent_type}<br>Reputation: {reputation:.2f}<br>Cash: ${cash:.0f}")
        node_color.append(color_map.get(agent_type, '#gray'))
        node_size.append(10 + reputation * 20)  # Size based on reputation
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.split('<br>')[0] for node in node_text],  # Show only agent ID as text
        hovertext=node_text,
        textposition="middle center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    ))
    
    fig.update_layout(
        title=dict(text="Agent Network", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Network shows alliances and trust relationships",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def plot_market_metrics(step_data):
    """Plot market performance metrics over time"""
    
    if not step_data:
        return go.Figure()
    
    df = pd.DataFrame(step_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Market Prices', 'Trading Volume', 'Agent Rewards', 'Market Efficiency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Market prices
    if 'market_prices' in df.columns:
        for resource in ResourceType:
            prices = [prices.get(resource, 0) for prices in df['market_prices']]
            fig.add_trace(
                go.Scatter(x=df['step'], y=prices, name=f"{resource.value} Price", mode='lines'),
                row=1, col=1
            )
    
    # Trading volume
    if 'trade_volume' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['trade_volume'], name='Volume', mode='lines+markers'),
            row=1, col=2
        )
    
    # Agent rewards
    if 'total_rewards' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['total_rewards'], name='Total Rewards', mode='lines'),
            row=2, col=1
        )
    
    # Market efficiency
    if 'market_efficiency' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['market_efficiency'], name='Efficiency', mode='lines'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True, title_text="Market Performance Metrics")
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤝 AI Negotiator Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Control Panel")
    
    # Load scenarios
    scenarios = load_scenarios()
    scenario_names = list(scenarios.keys()) if scenarios else ['test']
    
    selected_scenario = st.sidebar.selectbox(
        "Select Scenario",
        scenario_names,
        help="Choose a marketplace scenario to run"
    )
    
    # Scenario information
    if scenarios and selected_scenario in scenarios:
        scenario_config = scenarios[selected_scenario]
        st.sidebar.markdown("### Scenario Info")
        st.sidebar.markdown(f"**Name:** {scenario_config.get('name', 'N/A')}")
        st.sidebar.markdown(f"**Description:** {scenario_config.get('description', 'N/A')}")
        
        env_config = scenario_config.get('environment', {})
        st.sidebar.markdown(f"**Agents:** {env_config.get('num_agents', 'N/A')}")
        st.sidebar.markdown(f"**Max Steps:** {env_config.get('max_steps', 'N/A')}")
    
    # Simulation controls
    st.sidebar.markdown("### Simulation Controls")
    
    if st.sidebar.button("🚀 Initialize Simulation", disabled=st.session_state.simulation_running):
        with st.spinner("Initializing simulation..."):
            # Create environment
            if scenarios and selected_scenario in scenarios:
                env_config = scenarios[selected_scenario].get('environment', {})
                agent_config = scenarios[selected_scenario].get('agents', {})
                
                st.session_state.env = MarketplaceEnv(
                    num_agents=env_config.get('num_agents', 15),
                    max_steps=env_config.get('max_steps', 1000),
                    communication_enabled=env_config.get('communication_enabled', True),
                    alliance_enabled=env_config.get('alliance_enabled', True),
                    regulation_enabled=env_config.get('regulation_enabled', True)
                )
                
                # Create agents
                distribution = agent_config.get('distribution', {})
                total_agents = env_config.get('num_agents', 15)
                
                st.session_state.agents = AgentFactory.create_balanced_population(
                    total_agents=total_agents,
                    custom_distribution=distribution
                )
            else:
                # Default configuration
                st.session_state.env = MarketplaceEnv(num_agents=8, max_steps=200)
                st.session_state.agents = AgentFactory.create_balanced_population(8)
            
            # Reset environment
            obs, info = st.session_state.env.reset()
            st.session_state.step_data = []
            st.session_state.current_step = 0
            
            st.success(f"Simulation initialized with {len(st.session_state.agents)} agents!")
    
    # Step controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Step", disabled=not st.session_state.env):
            if st.session_state.env and st.session_state.agents:
                # Get current observations
                obs = st.session_state.env._get_observation(st.session_state.agents[0].agent_id) if st.session_state.agents else {}
                
                # Get actions from agents
                actions = {}
                for agent in st.session_state.agents:
                    try:
                        agent_obs = st.session_state.env._get_observation(agent.agent_id)
                        action = agent.get_action(agent_obs)
                        actions[agent.agent_id] = action
                    except:
                        # Fallback action
                        actions[agent.agent_id] = {
                            'trade_resource_type': 0,
                            'trade_quantity': [0],
                            'trade_price': [0],
                            'trade_target': 0,
                            'trade_action_type': 2,
                            'comm_enabled': 0,
                            'comm_message_type': 0,
                            'comm_target': 0,
                            'alliance_action': 0,
                            'alliance_target': 0
                        }
                
                # Step environment
                obs, rewards, terminations, truncations, infos = st.session_state.env.step(actions)
                
                # Collect step data
                step_info = {
                    'step': st.session_state.current_step,
                    'market_prices': dict(st.session_state.env.market_prices),
                    'trade_volume': len([t for t in st.session_state.env.trades.values() if t.status == 'completed']),
                    'total_rewards': sum(rewards.values()) if rewards else 0,
                    'market_efficiency': 0.7 + np.random.normal(0, 0.1),  # Simulated metric
                    'active_alliances': len(st.session_state.env.alliances),
                    'violations': sum(st.session_state.env.violation_counts.values())
                }
                
                st.session_state.step_data.append(step_info)
                st.session_state.current_step += 1
                
                st.success(f"Step {st.session_state.current_step} completed!")
    
    with col2:
        if st.button("⏹️ Reset", disabled=not st.session_state.env):
            if st.session_state.env:
                obs, info = st.session_state.env.reset()
                st.session_state.step_data = []
                st.session_state.current_step = 0
                st.success("Simulation reset!")
    
    # Auto-run toggle
    auto_run = st.sidebar.checkbox("🔄 Auto-run", help="Automatically step through simulation")
    
    if auto_run and st.session_state.env:
        time.sleep(1)  # 1 second delay
        st.rerun()
    
    # Main content area
    if not st.session_state.env:
        st.info("👆 Please initialize a simulation using the control panel")
        return
    
    # Current status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Step", st.session_state.current_step)
    
    with col2:
        active_trades = len([t for t in st.session_state.env.trades.values() if t.status == 'pending'])
        st.metric("Active Trades", active_trades)
    
    with col3:
        active_alliances = len(st.session_state.env.alliances)
        st.metric("Alliances", active_alliances)
    
    with col4:
        total_violations = sum(st.session_state.env.violation_counts.values())
        st.metric("Violations", total_violations)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "🤖 Agents", "📈 Performance", "🌐 Network"])
    
    with tab1:
        st.subheader("Market State")
        
        # Market prices
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Market Prices")
            price_data = []
            for resource_type, price in st.session_state.env.market_prices.items():
                price_data.append({
                    'Resource': resource_type.value,
                    'Price': f"${price:.2f}",
                    'Price_Value': price
                })
            
            if price_data:
                df_prices = pd.DataFrame(price_data)
                
                # Bar chart of prices
                fig_prices = px.bar(df_prices, x='Resource', y='Price_Value', 
                                  title='Current Market Prices')
                st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            st.markdown("#### Recent Trades")
            recent_trades = [t for t in st.session_state.env.trades.values() if t.timestamp > st.session_state.current_step - 10]
            
            if recent_trades:
                trade_data = []
                for trade in recent_trades[-5:]:  # Show last 5 trades
                    trade_data.append({
                        'Buyer': trade.buyer,
                        'Seller': trade.seller,
                        'Resource': trade.resource.type.value,
                        'Quantity': f"{trade.resource.quantity:.1f}",
                        'Price': f"${trade.agreed_price:.2f}",
                        'Status': trade.status
                    })
                
                df_trades = pd.DataFrame(trade_data)
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("No recent trades")
    
    with tab2:
        st.subheader("Agent Status")
        
        # Agent performance table
        agent_data = []
        for agent in st.session_state.agents:
            portfolio_value = sum(agent.portfolio.values()) if hasattr(agent, 'portfolio') else 0
            agent_data.append({
                'Agent ID': agent.agent_id,
                'Type': agent.agent_type.value,
                'Cash': f"${agent.cash:.0f}",
                'Portfolio Value': f"{portfolio_value:.1f}",
                'Reputation': f"{agent.reputation:.2f}",
                'Violations': st.session_state.env.violation_counts.get(agent.agent_id, 0)
            })
        
        df_agents = pd.DataFrame(agent_data)
        st.dataframe(df_agents, use_container_width=True)
        
        # Agent type distribution
        type_counts = df_agents['Type'].value_counts()
        fig_types = px.pie(values=type_counts.values, names=type_counts.index, 
                          title='Agent Type Distribution')
        st.plotly_chart(fig_types, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Metrics")
        
        if st.session_state.step_data:
            # Plot market metrics
            fig_metrics = plot_market_metrics(st.session_state.step_data)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Market Statistics")
                df_steps = pd.DataFrame(st.session_state.step_data)
                
                st.metric("Average Trading Volume", f"{df_steps['trade_volume'].mean():.1f}")
                st.metric("Market Efficiency", f"{df_steps['market_efficiency'].mean():.3f}")
                st.metric("Total Rewards", f"{df_steps['total_rewards'].sum():.1f}")
            
            with col2:
                st.markdown("#### Recent Trends")
                if len(df_steps) >= 2:
                    volume_trend = df_steps['trade_volume'].iloc[-1] - df_steps['trade_volume'].iloc[-2]
                    efficiency_trend = df_steps['market_efficiency'].iloc[-1] - df_steps['market_efficiency'].iloc[-2]
                    
                    st.metric("Volume Change", f"{volume_trend:+.1f}")
                    st.metric("Efficiency Change", f"{efficiency_trend:+.3f}")
        else:
            st.info("No performance data available yet. Run some simulation steps to see metrics.")
    
    with tab4:
        st.subheader("Agent Network")
        
        if st.session_state.agents:
            # Create and plot network graph
            G = create_agent_network_graph(st.session_state.agents, st.session_state.env.alliances)
            
            if G.number_of_nodes() > 0:
                fig_network = plot_network_graph(G)
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Network statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Connections", G.number_of_edges())
                
                with col2:
                    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                
                with col3:
                    st.metric("Network Density", f"{nx.density(G):.3f}")
            else:
                st.info("No network connections to display")
    
    # Footer
    st.markdown("---")
    st.markdown("**AI Negotiator Dashboard** | Multi-Agent Reinforcement Learning Marketplace")

if __name__ == "__main__":
    main()