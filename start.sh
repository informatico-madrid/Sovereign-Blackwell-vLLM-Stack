#!/bin/bash
# Sovereign AI Stack - Start/Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/deploy/compose/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"

# --- SAFETY CHECK ---
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found in $SCRIPT_DIR"
    echo "Please run: cp .env.example .env and configure it."
    exit 1
fi

case "$1" in
  start)
    echo "🚀 Starting Sovereign AI Stack..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    ;;
  stop)
    echo "🛑 Stopping Sovereign AI Stack..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    ;;
  restart)
    echo "🔄 Restarting Sovereign AI Stack..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    # Esperar un momento para liberar VRAM
    sleep 2
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "${2:-}"
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    ;;
  check)
    echo "🔍 Checking vLLM Health Status..."
    curl -s http://localhost:8000/health
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|logs [service]|status|check}"
    exit 1
    ;;
esac