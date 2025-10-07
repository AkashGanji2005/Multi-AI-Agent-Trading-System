#!/usr/bin/env python3
"""
Simple Test Dashboard for AI Negotiator
A lightweight version to test if Streamlit is working
"""

import streamlit as st
import sys
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page config
st.set_page_config(
    page_title="AI Negotiator - Test Dashboard",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤝 AI Negotiator - Test Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("Testing if the dashboard works!")
    
    # Test if we can import our modules
    try:
        from src.environment.marketplace import ResourceType, AgentType
        st.sidebar.success("✅ Core modules loaded")
        modules_loaded = True
    except Exception as e:
        st.sidebar.error(f"❌ Module loading failed: {e}")
        modules_loaded = False
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 System Status")
        
        # System status
        status_data = {
            "Component": ["Dashboard", "Core Modules", "Streamlit", "Python"],
            "Status": ["✅ Running", 
                      "✅ Loaded" if modules_loaded else "❌ Error",
                      "✅ Active", 
                      "✅ Working"],
            "Details": [
                "Port 8501",
                "AI Negotiator modules",
                f"Version {st.__version__}",
                f"Version {sys.version.split()[0]}"
            ]
        }
        
        df_status = pd.DataFrame(status_data)
        st.dataframe(df_status, use_container_width=True)
        
        # Quick test simulation button
        if st.button("🚀 Run Quick Test", type="primary"):
            if modules_loaded:
                with st.spinner("Running quick test..."):
                    try:
                        # Import and run a simple test
                        from src.agents.agent_factory import AgentFactory
                        from src.environment.marketplace import MarketplaceEnv
                        
                        # Create a small test
                        agents = AgentFactory.create_balanced_population(4)
                        env = MarketplaceEnv(num_agents=4, max_steps=10)
                        
                        st.success(f"✅ Test successful! Created {len(agents)} agents")
                        
                        # Show agent types
                        agent_types = {}
                        for agent in agents:
                            agent_type = agent.agent_type.value
                            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                        
                        st.write("**Agent Distribution:**")
                        for agent_type, count in agent_types.items():
                            st.write(f"- {agent_type.capitalize()}: {count}")
                            
                    except Exception as e:
                        st.error(f"❌ Test failed: {e}")
            else:
                st.error("Cannot run test - modules not loaded")
    
    with col2:
        st.header("📈 Sample Visualization")
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Energy Price': 100 + np.cumsum(np.random.randn(30) * 2),
            'Data Price': 50 + np.cumsum(np.random.randn(30) * 1.5),
            'Goods Price': 75 + np.cumsum(np.random.randn(30) * 1.8),
        })
        
        # Plot sample market data
        fig = px.line(
            sample_data, 
            x='Date', 
            y=['Energy Price', 'Data Price', 'Goods Price'],
            title="Sample Market Prices",
            labels={'value': 'Price ($)', 'variable': 'Resource Type'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample metrics
        st.subheader("📊 Sample Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Trades", "1,234", "+5.2%")
        
        with metric_col2:
            st.metric("Success Rate", "87.3%", "+2.1%")
        
        with metric_col3:
            st.metric("Avg Profit", "$45.67", "-1.4%")
    
    # Instructions
    st.markdown("---")
    st.header("🎯 How to Use")
    
    instructions_col1, instructions_col2 = st.columns(2)
    
    with instructions_col1:
        st.markdown("""
        **Dashboard Working!** 🎉
        
        If you can see this page, your Streamlit dashboard is working correctly.
        
        **Next Steps:**
        1. ✅ Dashboard is running on `http://localhost:8501`
        2. 🚀 Run a simulation: `python main_simple.py simulate`
        3. 📊 Use the full dashboard: `dashboard.py`
        4. 🎮 Try different scenarios
        """)
    
    with instructions_col2:
        st.markdown("""
        **Available Commands:**
        
        ```bash
        # Quick simulation
        python run_simulation.py
        
        # CLI with options
        python main_simple.py simulate --scenario energy_trading
        
        # List scenarios
        python main_simple.py scenarios
        
        # System info
        python main_simple.py info
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "AI Negotiator Dashboard Test | "
        f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()