#!/bin/bash
# Sovereign AI Stack - Start/Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/deploy/compose/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"
TEMPLATE_CONFIG="$SCRIPT_DIR/deploy/compose/litellm-config-template.yaml"

# --- SAFETY CHECK ---
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found in $SCRIPT_DIR"
    echo "Please run: cp .env.example .env and configure it."
    exit 1
fi

# Función para limpiar VRAM antes de iniciar
cleanup_vram() {
    echo "🧹 Limpiando VRAM y memoria compartida NCCL..."
    # Matar procesos relacionados con vLLM y NCCL
    pkill -f "vllm" 2>/dev/null || true
    pkill -f "python.*vllm" 2>/dev/null || true
    # Eliminar archivos de memoria compartida NCCL
    rm -rf /tmp/.nccl* 2>/dev/null || true
    # Esperar un momento para asegurar la limpieza
    sleep 1
}

# Función para generar config de LiteLLM dinámico
generate_litellm_config() {
    if [ ! -f "$TEMPLATE_CONFIG" ]; then
        echo "❌ Error: Plantilla de LiteLLM no encontrada en $TEMPLATE_CONFIG"
        exit 1
    fi
    
    # Generar archivo de configuración dinámico basado en SERVED_MODEL_NAME
    sed "s/\${SERVED_MODEL_NAME}/$SERVED_MODEL_NAME/g" "$TEMPLATE_CONFIG" > "$SCRIPT_DIR/deploy/compose/litellm-config.yaml"
}

case "$1" in
  start)
    echo "🚀 Iniciando Sovereign AI Stack..."
    
    # Verificar si se especificó un perfil
    if [ -n "$2" ] && [ "$2" != "default" ]; then
        PROFILE_PATH="$SCRIPT_DIR/deploy/profiles/$2.env"
        if [ -f "$PROFILE_PATH" ]; then
            echo "⚙️  Cargando perfil: $2"
            set -a
            # Primero el global
            source "$ENV_FILE"
            # Luego el perfil (sobrescribe lo anterior)
            source "$PROFILE_PATH"
            set +a
        else
            echo "❌ Error: Perfil $2 no encontrado."
            exit 1
        fi
    else
        # Cargar solo el .env global
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    # Limpiar VRAM antes de iniciar
    cleanup_vram
    
    # Generar configuración de LiteLLM dinámica
    generate_litellm_config
    
    # Iniciar servicios
    docker compose -f "$COMPOSE_FILE" up
    ;;
  stop)
    echo "🛑 Deteniendo Sovereign AI Stack..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  restart)
    echo "🔄 Reiniciando Sovereign AI Stack..."
    docker compose -f "$COMPOSE_FILE" down
    # Esperar un momento para liberar VRAM
    sleep 2
    # Limpiar VRAM antes de reiniciar
    cleanup_vram
    # Generar configuración de LiteLLM dinámica
    generate_litellm_config
    docker compose -f "$COMPOSE_FILE" up
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f "${2:-}"
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  check)
    echo "🔍 Verificando estado de salud de vLLM..."
    curl -s http://localhost:8000/health
    ;;
  *)
    echo "Uso: $0 {start|stop|restart|logs [service]|status|check}"
    echo "Para usar un perfil específico: $0 start <nombre_perfil>"
    exit 1
    ;;
esac