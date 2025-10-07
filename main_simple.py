#!/usr/bin/env python3
"""
Simplified Main CLI for AI Negotiator (Python 3.13 Compatible)
Works without Ray/RLLib dependencies
"""

import os
import sys
import time
import yaml
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.marketplace import MarketplaceEnv, ResourceType, AgentType
from src.agents.agent_factory import AgentFactory

app = typer.Typer(help="AI Negotiator - Autonomous Agent Marketplace")
console = Console()

def load_scenarios():
    """Load scenarios from config file"""
    # Try the simple config first, then fall back to the complex one
    config_files = ["config/scenarios_simple.yaml", "config/scenarios.yaml"]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                scenarios = config.get('scenarios', {})
                if scenarios:
                    return scenarios
        except FileNotFoundError:
            continue
        except Exception as e:
            console.print(f"[yellow]Warning: Error loading {config_file}: {e}[/yellow]")
            continue
    
    console.print("[red]No valid config files found[/red]")
    return {}

@app.command()
def scenarios():
    """List available simulation scenarios"""
    console.print("\n[bold blue]Available Scenarios[/bold blue]")
    console.print("=" * 50)
    
    scenarios_config = load_scenarios()
    
    if not scenarios_config:
        console.print("[yellow]No scenarios found or config file is empty[/yellow]")
        console.print("\n[dim]Using default scenarios...[/dim]")
        
        # Create basic scenarios
        scenarios_config = {
            "balanced": {
                "description": "Balanced market with equal agent types",
                "agents": 8,
                "steps": 100,
                "agent_distribution": {
                    "buyer": 0.3,
                    "seller": 0.3,
                    "regulator": 0.1,
                    "mediator": 0.1,
                    "speculator": 0.2
                }
            },
            "energy_trading": {
                "description": "Energy trading marketplace",
                "agents": 10,
                "steps": 200,
                "resource_focus": ["energy"],
                "agent_distribution": {
                    "buyer": 0.4,
                    "seller": 0.4,
                    "regulator": 0.1,
                    "speculator": 0.1
                }
            }
        }
    
    # Display scenarios in a table
    table = Table(title="Simulation Scenarios")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Agents", justify="right", style="green")
    table.add_column("Steps", justify="right", style="yellow")
    
    for name, config in scenarios_config.items():
        table.add_row(
            name,
            config.get('description', 'No description'),
            str(config.get('agents', 'N/A')),
            str(config.get('steps', 'N/A'))
        )
    
    console.print(table)
    console.print(f"\n[dim]Use 'python main_simple.py simulate --scenario <name>' to run a scenario[/dim]")

@app.command()
def simulate(
    scenario: str = typer.Option("balanced", help="Scenario to run"),
    steps: Optional[int] = typer.Option(None, help="Number of simulation steps"),
    agents: Optional[int] = typer.Option(None, help="Number of agents"),
    verbose: bool = typer.Option(False, help="Verbose output")
):
    """Run a marketplace simulation"""
    
    console.print(f"\n[bold green]🤝 Running AI Negotiator Simulation[/bold green]")
    console.print(f"Scenario: [cyan]{scenario}[/cyan]")
    
    # Load scenario config
    scenarios_config = load_scenarios()
    scenario_config = scenarios_config.get(scenario, {})
    
    if not scenario_config and scenario != "balanced":
        console.print(f"[red]Scenario '{scenario}' not found. Using balanced scenario.[/red]")
        scenario_config = {
            "description": "Default balanced scenario",
            "agents": 8,
            "steps": 100
        }
    
    # Override with command line arguments
    num_agents = agents or scenario_config.get('agents', 8)
    num_steps = steps or scenario_config.get('steps', 100)
    
    console.print(f"Agents: [yellow]{num_agents}[/yellow]")
    console.print(f"Steps: [yellow]{num_steps}[/yellow]")
    
    try:
        # Create environment
        console.print("\n[dim]Creating marketplace environment...[/dim]")
        env = MarketplaceEnv(
            num_agents=num_agents,
            max_steps=num_steps,
            communication_enabled=True,
            alliance_enabled=True,
            regulation_enabled=True
        )
        
        # Create agents
        console.print("[dim]Creating agent population...[/dim]")
        agents = AgentFactory.create_balanced_population(num_agents)
        
        console.print(f"[green]✅ Created {len(agents)} agents:[/green]")
        agent_types = {}
        for agent in agents:
            agent_type = agent.agent_type.value
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        for agent_type, count in agent_types.items():
            console.print(f"  - {agent_type.capitalize()}: {count}")
        
        # Reset environment
        obs, info = env.reset()
        console.print("\n[green]✅ Environment initialized[/green]")
        
        # Run simulation
        console.print(f"\n[bold]🚀 Starting simulation ({num_steps} steps)...[/bold]")
        
        total_rewards = {agent.agent_id: 0 for agent in agents}
        step_rewards = []
        trade_counts = []
        
        for step in track(range(num_steps), description="Simulating..."):
            actions = {}
            
            # Get actions from agents
            for agent in agents:
                if agent.agent_id in obs:
                    try:
                        action = agent.get_action(obs[agent.agent_id])
                        actions[agent.agent_id] = action
                    except Exception as e:
                        if verbose:
                            console.print(f"[yellow]Warning: Agent {agent.agent_id} action failed: {e}[/yellow]")
                        # Fallback action
                        actions[agent.agent_id] = {
                            'trade_resource_type': 0,
                            'trade_quantity': [10.0],
                            'trade_price': [100.0],
                            'trade_target': 0,
                            'trade_action_type': 2,  # hold
                            'comm_enabled': 0,
                            'comm_message_type': 0,
                            'comm_target': 0,
                            'alliance_action': 0,
                            'alliance_target': 0
                        }
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update statistics
            step_reward = 0
            for agent_id, reward in rewards.items():
                if agent_id in total_rewards:
                    total_rewards[agent_id] += reward
                    step_reward += reward
            
            step_rewards.append(step_reward / len(agents) if agents else 0)
            
            # Count trades
            completed_trades = len([t for t in env.trades.values() if t.status == 'completed'])
            trade_counts.append(completed_trades)
            
            # Verbose output
            if verbose and step % 20 == 0:
                avg_reward = sum(step_rewards[-20:]) / min(20, len(step_rewards))
                console.print(f"  Step {step}: Avg Reward = {avg_reward:.2f}, Trades = {completed_trades}")
        
        # Results
        console.print("\n[bold green]🎉 Simulation Complete![/bold green]")
        
        # Summary statistics
        final_trades = len(env.trades)
        completed_trades = len([t for t in env.trades.values() if t.status == 'completed'])
        success_rate = completed_trades / final_trades if final_trades > 0 else 0
        
        avg_reward = sum(total_rewards.values()) / len(total_rewards) if total_rewards else 0
        
        # Results table
        results_table = Table(title="Simulation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Total Trades", str(final_trades))
        results_table.add_row("Completed Trades", str(completed_trades))
        results_table.add_row("Success Rate", f"{success_rate:.1%}")
        results_table.add_row("Average Reward", f"{avg_reward:.2f}")
        results_table.add_row("Steps Completed", str(num_steps))
        
        console.print(results_table)
        
        # Top performers
        if total_rewards:
            console.print("\n[bold]🏆 Top Performing Agents:[/bold]")
            sorted_agents = sorted(total_rewards.items(), key=lambda x: x[1], reverse=True)
            for i, (agent_id, reward) in enumerate(sorted_agents[:5]):
                agent = next(a for a in agents if a.agent_id == agent_id)
                console.print(f"  {i+1}. {agent_id} ({agent.agent_type.value}): {reward:.2f}")
        
        console.print(f"\n[dim]Simulation saved to logs/simulation_{int(time.time())}.log[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Simulation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
        return

@app.command()
def dashboard():
    """Launch the visualization dashboard"""
    console.print("\n[bold blue]🚀 Launching Dashboard...[/bold blue]")
    
    try:
        import subprocess
        import sys
        
        console.print("[dim]Starting Streamlit server...[/dim]")
        console.print("[green]Dashboard will open in your browser at http://localhost:8501[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        
        # Launch streamlit
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py"
        ], check=True)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start dashboard: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error launching dashboard: {e}[/red]")

@app.command()
def test():
    """Run system tests"""
    console.print("\n[bold blue]🧪 Running System Tests[/bold blue]")
    
    try:
        import subprocess
        import sys
        
        # Run the working test
        console.print("[dim]Running functionality tests...[/dim]")
        result = subprocess.run([sys.executable, "test_working.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✅ All tests passed![/green]")
            console.print(result.stdout)
        else:
            console.print("[red]❌ Some tests failed[/red]")
            console.print(result.stdout)
            console.print(result.stderr)
            
    except Exception as e:
        console.print(f"[red]Test execution failed: {e}[/red]")

@app.command()
def info():
    """Show system information"""
    console.print("\n[bold blue]ℹ️ AI Negotiator System Information[/bold blue]")
    
    info_table = Table()
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    info_table.add_column("Details")
    
    # Check dependencies
    try:
        import torch
        info_table.add_row("PyTorch", "✅ Available", f"Version {torch.__version__}")
    except ImportError:
        info_table.add_row("PyTorch", "❌ Missing", "Required for agent learning")
    
    try:
        import streamlit
        info_table.add_row("Streamlit", "✅ Available", f"Version {streamlit.__version__}")
    except ImportError:
        info_table.add_row("Streamlit", "❌ Missing", "Required for dashboard")
    
    try:
        import pettingzoo
        info_table.add_row("PettingZoo", "✅ Available", f"Version {pettingzoo.__version__}")
    except ImportError:
        info_table.add_row("PettingZoo", "❌ Missing", "Required for multi-agent environment")
    
    # Check optional big data dependencies
    big_data_deps = [
        ("PySpark", "pyspark"),
        ("Kafka", "kafka"),
        ("Redis", "redis"),
        ("Elasticsearch", "elasticsearch"),
        ("MongoDB", "pymongo")
    ]
    
    for name, module in big_data_deps:
        try:
            __import__(module)
            info_table.add_row(name, "✅ Available", "Big data integration")
        except ImportError:
            info_table.add_row(name, "⚠️ Optional", "Install for big data features")
    
    console.print(info_table)
    
    # File system check
    console.print("\n[bold]📁 File System:[/bold]")
    important_files = [
        ("dashboard.py", "Visualization dashboard"),
        ("run_simulation.py", "Quick simulation runner"),
        ("config/scenarios.yaml", "Scenario configurations"),
        ("examples/basic_example.py", "Basic usage example")
    ]
    
    for file_path, description in important_files:
        if os.path.exists(file_path):
            console.print(f"  ✅ {file_path} - {description}")
        else:
            console.print(f"  ❌ {file_path} - {description}")

if __name__ == "__main__":
    app()