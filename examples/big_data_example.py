#!/usr/bin/env python3
"""
Big Data Integration Example for AI Negotiator
Demonstrates how to use Apache Spark, Kafka, and other big data frameworks
"""

import os
import sys
import time
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.big_data_integration import BigDataMarketAnalytics, BigDataConfig
from src.data.spark_processor import SparkMarketDataProcessor
from src.data.kafka_streaming import KafkaMarketStreamer, MarketDataPoint, AgentActionEvent
from src.environment.marketplace import MarketplaceEnv, ResourceType, AgentType
from src.agents.agent_factory import AgentFactory

def demonstrate_spark_analytics():
    """Demonstrate Apache Spark for large-scale analytics"""
    
    print("\n" + "="*60)
    print("🔥 APACHE SPARK ANALYTICS DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize Spark processor
        spark_processor = SparkMarketDataProcessor(
            app_name="AI_Negotiator_Demo",
            master="local[4]"  # Use 4 cores
        )
        
        print("✓ Spark initialized successfully")
        
        # Create sample market data (simulating 1 million records)
        print("📊 Generating large-scale market data...")
        
        import numpy as np
        import pandas as pd
        
        # Generate 100,000 sample trades
        n_trades = 100000
        sample_data = []
        
        for i in range(n_trades):
            trade_data = {
                "trade_id": f"trade_{i}",
                "timestamp": datetime.now() - timedelta(hours=i//1000),
                "buyer_id": f"buyer_{i % 100}",
                "seller_id": f"seller_{i % 80}",
                "resource_type": np.random.choice(["energy", "data", "goods", "services"]),
                "quantity": np.random.uniform(1, 1000),
                "agreed_price": np.random.uniform(10, 500),
                "market_price": np.random.uniform(10, 500),
                "quality": np.random.uniform(0.3, 1.0),
                "status": np.random.choice(["completed", "failed"], p=[0.85, 0.15]),
                "buyer_type": "buyer",
                "seller_type": "seller"
            }
            sample_data.append(trade_data)
        
        # Convert to Spark DataFrame
        trade_df = spark_processor.spark.createDataFrame(sample_data, spark_processor.trade_schema)
        print(f"✓ Created Spark DataFrame with {trade_df.count():,} trades")
        
        # Cache for performance
        trade_df.cache()
        
        # 1. Market Trend Analysis
        print("\n📈 Analyzing market trends...")
        trends = spark_processor.analyze_market_trends(trade_df, window_hours=48)
        trend_results = trends.collect()
        
        print(f"✓ Analyzed trends for {len(trend_results)} time periods")
        for trend in trend_results[:5]:  # Show first 5
            print(f"   {trend.resource_type} @ {trend.hour}: "
                  f"Avg Price: ${trend.avg_price:.2f}, "
                  f"Volume: {trend.total_volume:.0f}, "
                  f"Success Rate: {trend.success_rate:.2%}")
        
        # 2. Market Anomaly Detection
        print("\n🚨 Detecting market anomalies...")
        
        # Create market data from trades
        market_data = trade_df.select(
            "timestamp", "resource_type", 
            spark_processor.spark.sql.functions.col("agreed_price").alias("price"),
            "quantity"
        ).groupBy("timestamp", "resource_type").agg(
            spark_processor.spark.sql.functions.avg("price").alias("price"),
            spark_processor.spark.sql.functions.sum("quantity").alias("volume")
        ).withColumn("volatility", spark_processor.spark.sql.functions.lit(0.15)) \
         .withColumn("bid_ask_spread", spark_processor.spark.sql.functions.lit(2.0)) \
         .withColumn("market_depth", spark_processor.spark.sql.functions.lit(100.0))
        
        anomalies = spark_processor.detect_market_anomalies(market_data, z_threshold=2.5)
        anomaly_count = anomalies.count()
        
        print(f"✓ Detected {anomaly_count} market anomalies")
        if anomaly_count > 0:
            anomaly_results = anomalies.collect()
            for anomaly in anomaly_results[:3]:  # Show first 3
                print(f"   {anomaly.resource_type}: "
                      f"Price: ${anomaly.price:.2f}, "
                      f"Z-Score: {anomaly.z_score:.2f}, "
                      f"Type: {anomaly.anomaly_type}")
        
        # 3. Agent Behavior Analysis
        print("\n🤖 Analyzing agent behaviors...")
        
        # Create agent behavior data
        behavior_data = []
        for i in range(10000):  # 10k behavior records
            behavior_data.append({
                "timestamp": datetime.now() - timedelta(minutes=i),
                "agent_id": f"agent_{i % 50}",
                "agent_type": np.random.choice(["buyer", "seller", "speculator", "mediator"]),
                "action_type": np.random.choice(["trade", "communicate", "alliance"]),
                "resource_type": np.random.choice(["energy", "data", "goods", "services"]),
                "quantity": np.random.uniform(1, 100),
                "price": np.random.uniform(10, 200),
                "reward": np.random.normal(5, 10),
                "reputation": np.random.uniform(0.2, 1.0),
                "cash": np.random.uniform(100, 10000),
                "portfolio_value": np.random.uniform(500, 5000)
            })
        
        behavior_df = spark_processor.spark.createDataFrame(behavior_data, spark_processor.agent_behavior_schema)
        
        # Calculate performance metrics
        performance = spark_processor.calculate_agent_performance_metrics(behavior_df)
        performance_results = performance.collect()
        
        print(f"✓ Analyzed performance for {len(performance_results)} agents")
        
        # Show top performers
        top_performers = sorted(performance_results, key=lambda x: x.avg_reward, reverse=True)[:5]
        print("   Top performing agents:")
        for agent in top_performers:
            print(f"   - {agent.agent_id} ({agent.agent_type}): "
                  f"Avg Reward: {agent.avg_reward:.2f}, "
                  f"Actions: {agent.total_actions}")
        
        # 4. Agent Clustering
        print("\n🎯 Clustering agent behaviors...")
        clustered_agents, model = spark_processor.cluster_agent_behaviors(behavior_df, k=4)
        cluster_results = clustered_agents.collect()
        
        print(f"✓ Clustered {len(cluster_results)} agents into 4 groups")
        
        # Show cluster distribution
        cluster_counts = {}
        for agent in cluster_results:
            cluster_counts[agent.cluster] = cluster_counts.get(agent.cluster, 0) + 1
        
        for cluster_id, count in cluster_counts.items():
            print(f"   Cluster {cluster_id}: {count} agents")
        
        # 5. Export Results
        print("\n💾 Exporting results to data lake...")
        
        # Export to Parquet format
        output_path = "data/spark_analysis_results"
        
        spark_processor.export_to_data_lake(trends, f"{output_path}/market_trends", "parquet")
        spark_processor.export_to_data_lake(performance, f"{output_path}/agent_performance", "parquet")
        spark_processor.export_to_data_lake(clustered_agents, f"{output_path}/agent_clusters", "parquet")
        
        print("✓ Results exported to Parquet format")
        
        # Performance metrics
        print(f"\n📊 Processing Statistics:")
        print(f"   - Total trades processed: {n_trades:,}")
        print(f"   - Spark partitions used: {trade_df.rdd.getNumPartitions()}")
        print(f"   - Available cores: {spark_processor.spark.sparkContext.defaultParallelism}")
        
        return spark_processor
        
    except Exception as e:
        print(f"❌ Spark demonstration error: {e}")
        return None

def demonstrate_kafka_streaming():
    """Demonstrate Apache Kafka for real-time streaming"""
    
    print("\n" + "="*60)
    print("⚡ APACHE KAFKA STREAMING DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize Kafka streamer
        kafka_streamer = KafkaMarketStreamer(
            bootstrap_servers="localhost:9092"
        )
        
        print("✓ Kafka streamer initialized")
        
        # Start streaming
        kafka_streamer.start_streaming()
        print("✓ Kafka streaming started")
        
        # Register message handlers
        message_count = 0
        
        def market_data_handler(topic: str, data: dict):
            nonlocal message_count
            message_count += 1
            if message_count <= 5:  # Show first 5 messages
                print(f"📊 Market Data: {data['resource_type']} @ ${data['price']:.2f}")
        
        def trade_handler(topic: str, data: dict):
            print(f"💼 Trade: {data['buyer_id']} -> {data['seller_id']} "
                  f"({data['quantity']:.1f} {data['resource_type']})")
        
        kafka_streamer.register_handler('market_data', market_data_handler)
        kafka_streamer.register_handler('trade_events', trade_handler)
        
        print("✓ Message handlers registered")
        
        # Simulate real-time data publishing
        print("\n🚀 Publishing real-time market data...")
        
        import random
        
        for i in range(20):
            # Publish market data
            market_data = MarketDataPoint(
                timestamp=time.time(),
                resource_type=random.choice(['energy', 'data', 'goods', 'services']),
                price=random.uniform(50, 200),
                volume=random.uniform(100, 1000),
                volatility=random.uniform(0.05, 0.3),
                bid_ask_spread=random.uniform(1, 5),
                market_depth=random.uniform(50, 500),
                num_active_traders=random.randint(10, 100)
            )
            
            kafka_streamer.publish_market_data(market_data)
            
            # Publish agent action
            agent_action = AgentActionEvent(
                timestamp=time.time(),
                agent_id=f"agent_{random.randint(1, 20)}",
                agent_type=random.choice(['buyer', 'seller', 'speculator']),
                action_type='trade',
                resource_type=market_data.resource_type,
                quantity=random.uniform(1, 50),
                price=market_data.price,
                success=random.choice([True, False]),
                reward=random.uniform(-5, 15)
            )
            
            kafka_streamer.publish_agent_action(agent_action)
            
            # Publish system alert occasionally
            if i % 5 == 0:
                kafka_streamer.publish_alert(
                    alert_type="market_activity",
                    severity="info",
                    message=f"High activity detected in {market_data.resource_type} market",
                    metadata={"price": market_data.price, "volume": market_data.volume}
                )
            
            time.sleep(0.2)  # 200ms between messages
        
        print(f"✓ Published {20} market data points and agent actions")
        
        # Wait for message processing
        time.sleep(2)
        
        # Get topic statistics
        stats = kafka_streamer.get_topic_stats()
        print(f"\n📈 Kafka Topic Statistics:")
        for topic, info in stats.items():
            print(f"   {topic}: {info['partitions']} partitions")
        
        return kafka_streamer
        
    except Exception as e:
        print(f"❌ Kafka demonstration error: {e}")
        print("   Note: Make sure Kafka is running on localhost:9092")
        return None

def demonstrate_integrated_pipeline():
    """Demonstrate integrated big data pipeline"""
    
    print("\n" + "="*60)
    print("🔗 INTEGRATED BIG DATA PIPELINE DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize integrated analytics
        config = BigDataConfig(
            spark_master="local[*]",
            kafka_bootstrap_servers="localhost:9092",
            redis_host="localhost",
            elasticsearch_hosts=["localhost:9200"],
            mongodb_host="localhost"
        )
        
        analytics = BigDataMarketAnalytics(config)
        print("✓ Integrated analytics platform initialized")
        
        # Start real-time processing
        analytics.start_real_time_processing()
        print("✓ Real-time processing pipeline started")
        
        # Simulate marketplace activity
        print("\n🏪 Simulating marketplace activity...")
        
        # Create a small marketplace simulation
        env = MarketplaceEnv(num_agents=8, max_steps=50)
        agents = AgentFactory.create_balanced_population(8)
        
        obs, info = env.reset()
        
        for step in range(50):
            actions = {}
            
            # Get actions from agents
            for agent in agents:
                if agent.agent_id in obs:
                    try:
                        action = agent.get_action(obs[agent.agent_id])
                        actions[agent.agent_id] = action
                    except:
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
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Publish data to streaming pipeline
            if analytics.kafka_streamer:
                # Publish market data
                for resource_type, price in env.market_prices.items():
                    market_data = MarketDataPoint(
                        timestamp=time.time(),
                        resource_type=resource_type.value,
                        price=price,
                        volume=len([t for t in env.trades.values() if t.resource.type == resource_type]),
                        volatility=0.1,
                        bid_ask_spread=2.0,
                        market_depth=100.0,
                        num_active_traders=len(agents)
                    )
                    analytics.kafka_streamer.publish_market_data(market_data)
                
                # Publish agent actions
                for agent_id, reward in rewards.items():
                    if agent_id in actions:
                        agent = next(a for a in agents if a.agent_id == agent_id)
                        action_event = AgentActionEvent(
                            timestamp=time.time(),
                            agent_id=agent_id,
                            agent_type=agent.agent_type.value,
                            action_type='trade',
                            resource_type='energy',
                            quantity=10.0,
                            price=100.0,
                            success=reward > 0,
                            reward=reward
                        )
                        analytics.kafka_streamer.publish_agent_action(action_event)
            
            if step % 10 == 0:
                print(f"   Step {step}: {len([t for t in env.trades.values() if t.status == 'completed'])} trades completed")
        
        print("✓ Marketplace simulation completed")
        
        # Wait for data processing
        time.sleep(3)
        
        # Run batch analysis
        print("\n📊 Running comprehensive batch analysis...")
        batch_results = analytics.run_batch_analysis("comprehensive")
        print(f"✓ Batch analysis completed: {len(batch_results)} result sets")
        
        # Get real-time dashboard data
        dashboard_data = analytics.get_real_time_dashboard_data()
        print(f"\n📱 Dashboard data retrieved: {len(dashboard_data)} sections")
        
        # Search for events (if Elasticsearch is available)
        if analytics.elasticsearch_client:
            search_results = analytics.search_market_events("energy OR high", time_range="1h")
            print(f"🔍 Found {len(search_results)} events matching search criteria")
        
        # Get system status
        status = analytics.get_system_status()
        print(f"\n🖥️ System Status:")
        for system, available in status['systems'].items():
            status_icon = "✓" if available else "✗"
            print(f"   {system.capitalize()}: {status_icon}")
        
        # Export to data warehouse
        print(f"\n🏢 Exporting to data warehouse...")
        export_success = analytics.export_data_warehouse("parquet", "data_warehouse")
        if export_success:
            print("✓ Data warehouse export completed")
        else:
            print("⚠️ Data warehouse export failed (some systems may not be available)")
        
        return analytics
        
    except Exception as e:
        print(f"❌ Integrated pipeline error: {e}")
        return None

def main():
    """Main demonstration function"""
    
    print("🚀 AI NEGOTIATOR - BIG DATA INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates how to integrate big data frameworks")
    print("with the AI Negotiator marketplace for large-scale analytics.")
    print("=" * 80)
    
    # Check available systems
    print("\n🔍 Checking available big data systems...")
    
    systems_status = {
        "Apache Spark": True,  # Always available with PySpark
        "Apache Kafka": False,  # Requires running Kafka server
        "Redis": False,         # Requires running Redis server
        "Elasticsearch": False, # Requires running Elasticsearch
        "MongoDB": False,       # Requires running MongoDB
        "Cassandra": False      # Requires running Cassandra
    }
    
    # Try to detect running services
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        systems_status["Redis"] = True
    except:
        pass
    
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        producer.close()
        systems_status["Apache Kafka"] = True
    except:
        pass
    
    for system, available in systems_status.items():
        status_icon = "✓" if available else "✗"
        print(f"   {system}: {status_icon}")
    
    print(f"\nNote: Systems marked with ✗ require separate installation and setup.")
    print(f"The demonstration will work with whatever systems are available.\n")
    
    # Run demonstrations
    results = {}
    
    # 1. Spark Analytics (always available)
    spark_processor = demonstrate_spark_analytics()
    results['spark'] = spark_processor is not None
    
    # 2. Kafka Streaming (if available)
    if systems_status["Apache Kafka"]:
        kafka_streamer = demonstrate_kafka_streaming()
        results['kafka'] = kafka_streamer is not None
        
        if kafka_streamer:
            kafka_streamer.stop_streaming()
    else:
        print("\n⚠️ Skipping Kafka demonstration (Kafka not running)")
        results['kafka'] = False
    
    # 3. Integrated Pipeline
    analytics = demonstrate_integrated_pipeline()
    results['integrated'] = analytics is not None
    
    # Cleanup
    if spark_processor:
        spark_processor.close()
    
    if analytics:
        analytics.close()
    
    # Summary
    print("\n" + "="*80)
    print("📋 DEMONSTRATION SUMMARY")
    print("="*80)
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"Successfully completed {successful_demos}/{total_demos} demonstrations:")
    
    for demo, success in results.items():
        status_icon = "✅" if success else "❌"
        demo_name = {
            'spark': 'Apache Spark Analytics',
            'kafka': 'Apache Kafka Streaming', 
            'integrated': 'Integrated Big Data Pipeline'
        }[demo]
        print(f"   {status_icon} {demo_name}")
    
    if successful_demos == total_demos:
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"   Your big data integration is fully functional.")
    else:
        print(f"\n⚠️ Some demonstrations were skipped due to missing services.")
        print(f"   Install and start the required services for full functionality.")
    
    print(f"\n💡 Next Steps:")
    print(f"   - Install missing big data services for full functionality")
    print(f"   - Customize the BigDataConfig for your infrastructure")
    print(f"   - Integrate with your existing data pipeline")
    print(f"   - Scale up with distributed computing clusters")
    
    print(f"\n🔗 Integration Benefits:")
    print(f"   - Real-time market monitoring and alerts")
    print(f"   - Large-scale historical data analysis")
    print(f"   - Distributed agent training and simulation")
    print(f"   - Advanced market prediction and optimization")

if __name__ == "__main__":
    main()