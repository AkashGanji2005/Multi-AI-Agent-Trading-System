# AI Negotiator Big Data Setup Script (PowerShell)
# Sets up all required big data infrastructure using Docker

Write-Host "🚀 AI Negotiator - Big Data Infrastructure Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is installed
try {
    docker-compose --version | Out-Null
    Write-Host "✅ Docker Compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "📁 Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\exports" | Out-Null
New-Item -ItemType Directory -Force -Path "notebooks" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "✅ Directories created" -ForegroundColor Green

# Start the big data infrastructure
Write-Host "🐳 Starting big data infrastructure..." -ForegroundColor Yellow
Write-Host "This may take several minutes on first run..." -ForegroundColor Yellow

docker-compose up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service health
Write-Host "🔍 Checking service health..." -ForegroundColor Yellow

$services = @{
    "Kafka" = 9092
    "Redis" = 6379
    "Elasticsearch" = 9200
    "MongoDB" = 27017
    "Cassandra" = 9042
}

$healthyServices = 0

foreach ($service in $services.GetEnumerator()) {
    $name = $service.Key
    $port = $service.Value
    
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.ConnectAsync("localhost", $port).Wait(1000)
        if ($connection.Connected) {
            Write-Host "✅ $name is running on port $port" -ForegroundColor Green
            $healthyServices++
            $connection.Close()
        } else {
            Write-Host "❌ $name is not responding on port $port" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ $name is not responding on port $port" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📊 Service Status: $healthyServices/$($services.Count) services are healthy" -ForegroundColor Cyan

if ($healthyServices -eq $services.Count) {
    Write-Host "🎉 All services are running successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some services may still be starting up. Check with 'docker-compose ps'" -ForegroundColor Yellow
}

# Display access information
Write-Host ""
Write-Host "🌐 Service Access Information:" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Kafka:          localhost:9092"
Write-Host "Redis:          localhost:6379"
Write-Host "Elasticsearch:  http://localhost:9200"
Write-Host "Kibana:         http://localhost:5601"
Write-Host "MongoDB:        localhost:27017 (admin/password)"
Write-Host "Cassandra:      localhost:9042"
Write-Host "Spark Master:   http://localhost:8080"
Write-Host "Spark Worker:   http://localhost:8081"
Write-Host "Jupyter:        http://localhost:8888"
Write-Host "Grafana:        http://localhost:3000 (admin/admin)"
Write-Host "InfluxDB:       http://localhost:8086 (admin/password)"

Write-Host ""
Write-Host "🔧 Next Steps:" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "1. Install Python dependencies:"
Write-Host "   pip install -r requirements.txt"
Write-Host ""
Write-Host "2. Test the big data integration:"
Write-Host "   python examples/big_data_example.py"
Write-Host ""
Write-Host "3. Run the marketplace with big data:"
Write-Host "   python main.py train --scenario energy_trading --wandb-project ai-negotiator-big-data"
Write-Host ""
Write-Host "4. Access Jupyter notebooks for analysis:"
Write-Host "   Open http://localhost:8888 in your browser"
Write-Host ""
Write-Host "5. Monitor with Grafana:"
Write-Host "   Open http://localhost:3000 (admin/admin)"

Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "=================="
Write-Host "- Kafka Topics: market_data, trade_events, agent_actions, communications, alerts, analytics"
Write-Host "- Redis Keys: market:*, agent:*, analysis:*"
Write-Host "- MongoDB Database: ai_negotiator"
Write-Host "- Cassandra Keyspace: ai_negotiator"
Write-Host "- Elasticsearch Indices: ai_negotiator-*"

Write-Host ""
Write-Host "🛑 To stop all services:" -ForegroundColor Yellow
Write-Host "docker-compose down"
Write-Host ""
Write-Host "🗑️ To remove all data (CAUTION - this will delete all stored data):" -ForegroundColor Red
Write-Host "docker-compose down -v"

Write-Host ""
Write-Host "✨ Big data infrastructure setup complete!" -ForegroundColor Green