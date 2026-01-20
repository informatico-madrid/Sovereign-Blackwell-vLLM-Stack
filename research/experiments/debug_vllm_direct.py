import requests
import json

# CONFIGURACIÓN: Apuntamos al motor directamente (bypass LiteLLM)
VLLM_URL = "http://localhost:8000/v1/chat/completions"

def inject_to_vllm():
    # 1. Cargar el JSON que ya tienes limpio y depurado
    try:
        with open("debug_input.json", "r") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"❌ Error cargando debug_input.json: {e}")
        return

    # 2. Forzamos parámetros para ver qué hace el modelo
    # 'model' debe coincidir con el nombre que le diste en vLLM (qwen-bunker)
    payload["model"] = "qwen-bunker" 
    payload["temperature"] = 0 # Queremos determinismo puro para debug
    payload["stream"] = False   # En debug es mejor ver el bloque completo

    print(f"--- 🚀 Inyectando prompt al motor Blackwell ({VLLM_URL}) ---")
    
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']
            
            print("\n" + "="*50)
            print("🎬 LA OTRA MITAD DE LA PELÍCULA (Respuesta de la GPU)")
            print("="*50)
            
            # 1. Qué texto escribió el modelo (Aquí veremos si puso las etiquetas <tool_call>)
            print(f"\n[TEXTO CRUDO (content)]:\n{message.get('content')}")
            
            # 2. Qué entendió el Parser de vLLM (Aquí veremos si la transformación funcionó)
            print(f"\n[LLAMADAS A HERRAMIENTAS (tool_calls)]:")
            if message.get('tool_calls'):
                print(json.dumps(message['tool_calls'], indent=2, ensure_ascii=False))
            else:
                print("⚠️  NULL: El parser nativo NO detectó herramientas en el texto.")
                
            print("\n" + "="*50)
            print(f"Tokens Prompt: {result['usage']['prompt_tokens']}")
            print(f"Tokens Gen: {result['usage']['completion_tokens']}")
            
        else:
            print(f"❌ Error del motor: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Error de conexión: {e}")

if __name__ == "__main__":
    inject_to_vllm()