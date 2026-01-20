# 🚀 Upgrade a Qwen3-Coder-30B-A3B

## Estado de descarga
Modelo descargándose en: `/mnt/bunker_data/ai/models/qwen3-coder-30b-moe`

## Características del nuevo modelo

| Feature | Qwen2.5-Coder-32B | Qwen3-Coder-30B-A3B |
|---------|-------------------|---------------------|
| Contexto nativo | 32K | **256K** |
| Contexto extendido | ~65K (problemático) | **1M con YaRN** |
| Arquitectura | Dense 32B | MoE 30B (3.3B activos) |
| Velocidad | ~15 tok/s | **~100+ tok/s** |
| SWE-bench | 50.8% | **69.6%** |
| Thinking mode | Sí (forzado) | **No necesario** |

## Cambios necesarios en la configuración

### 1. .env
```bash
MODELS_PATH=${MODELS_ROOT}/qwen3-coder-30b-moe
SERVED_MODEL_NAME=qwen3-coder-30b-moe
MAX_MODEL_LEN=262144  # 256K nativo!
```

### 2. docker-compose.yml
```yaml
--quantization awq  # Cambiar si no es AWQ
--temperature 0.7
--top-p 0.8
--top-k 20
--repetition-penalty 1.05
```

### 3. Template Jinja
- El modelo NO usa `<think>` tags
- Tiene tool calling nativo optimizado
- Posiblemente usar template por defecto del modelo

### 4. Parser
- Puede que funcione el parser actual
- O usar el parser nativo de vLLM para Qwen3

## Comando para verificar descarga
```bash
ls -la /mnt/bunker_data/ai/models/qwen3-coder-30b-moe/
du -sh /mnt/bunker_data/ai/models/qwen3-coder-30b-moe/
```

## Comando para cambiar al nuevo modelo
```bash
# 1. Actualizar .env
sed -i 's|qwen2.5-coder-32b-awq|qwen3-coder-30b-moe|g' .env
sed -i 's/MAX_MODEL_LEN=.*/MAX_MODEL_LEN=262144/' .env

# 2. Reiniciar
docker compose up -d vllm-engine
```
