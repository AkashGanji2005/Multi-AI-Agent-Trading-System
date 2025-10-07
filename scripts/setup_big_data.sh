#!/bin/bash

# AI Negotiator Big Data Setup Script
# Sets up all required big data infrastructure using Docker

echo "🚀 AI Negotiator - Big Data Infrastructure Setup"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/processed
mkdir -p data/raw
mkdir -p data/exports
mkdir -p notebooks
mkdir -p logs

echo "✅ Directories created"

# Start the big data infrastructure
echo "🐳 Starting big data infrastructure..."
echo "This may take several minutes on first run..."

docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=("kafka:9092" "redis:6379" "elasticsearch:9200" "mongodb:27017" "cassandra:9042")
healthy_services=0

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $name is running on port $port"
        ((healthy_services++))
    else
        echo "❌ $name is not responding on port $port"
    fi
done

echo ""
echo "📊 Service Status: $healthy_services/${#services[@]} services are healthy"

if [ $healthy_services -eq ${#services[@]} ]; then
    echo "🎉 All services are running successfully!"
else
    echo "⚠️  Some services may still be starting up. Check with 'docker-compose ps'"
fi

# Display access information
echo ""
echo "🌐 Service Access Information:"
echo "================================"
echo "Kafka:          localhost:9092"
echo "Redis:          localhost:6379"
echo "Elasticsearch:  http://localhost:9200"
echo "Kibana:         http://localhost:5601"
echo "MongoDB:        localhost:27017 (admin/password)"
echo "Cassandra:      localhost:9042"
echo "Spark Master:   http://localhost:8080"
echo "Spark Worker:   http://localhost:8081"
echo "Jupyter:        http://localhost:8888"
echo "Grafana:        http://localhost:3000 (admin/admin)"
echo "InfluxDB:       http://localhost:8086 (admin/password)"

echo ""
echo "🔧 Next Steps:"
echo "==============="
echo "1. Install Python dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Test the big data integration:"
echo "   python examples/big_data_example.py"
echo ""
echo "3. Run the marketplace with big data:"
echo "   python main.py train --scenario energy_trading --wandb-project ai-negotiator-big-data"
echo ""
echo "4. Access Jupyter notebooks for analysis:"
echo "   Open http://localhost:8888 in your browser"
echo ""
echo "5. Monitor with Grafana:"
echo "   Open http://localhost:3000 (admin/admin)"

echo ""
echo "📖 Documentation:"
echo "=================="
echo "- Kafka Topics: market_data, trade_events, agent_actions, communications, alerts, analytics"
echo "- Redis Keys: market:*, agent:*, analysis:*"
echo "- MongoDB Database: ai_negotiator"
echo "- Cassandra Keyspace: ai_negotiator"
echo "- Elasticsearch Indices: ai_negotiator-*"

echo ""
echo "🛑 To stop all services:"
echo "docker-compose down"
echo ""
echo "🗑️ To remove all data (CAUTION - this will delete all stored data):"
echo "docker-compose down -v"

echo ""
echo "✨ Big data infrastructure setup complete!"