import requests
import time
import json

URL = "http://localhost:4000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer sk-master-bunker-2026", "Content-Type": "application/json"}

# Simulamos un contexto de código legacy masivo (~20k-30k tokens para empezar)
# Inyectamos una estructura de clases repetitiva para simular un codebase grande
legacy_code_chunk = """
class LegacyOrderProcessor {
    public function process($order) {
        // Lógica redundante para inflar el contexto
        if (true) { echo "Processing..."; }
    }
}
""" * 500 

prompt = f"Aquí tienes el código de mi sistema legacy:\n{legacy_code_chunk}\n\nPregunta: ¿Qué hace la clase LegacyOrderProcessor y cómo mejorarías su escalabilidad?"

payload = {
    "model": "bunker-agent",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
    "max_tokens": 100
}

print("🔥 Enviando 20K+ tokens de contexto a las 5090...")
start_time = time.time()
try:
    response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=300)
    end_time = time.time()
    
    if response.status_code == 200:
        res_json = response.json()
        duration = end_time - start_time
        prompt_t = res_json['usage']['prompt_tokens']
        comp_t = res_json['usage']['completion_tokens']
        
        print(f"\n--- RESULTADOS DEL STRESS TEST ---")
        print(f"Tokens de entrada (Prompt): {prompt_t}")
        print(f"Tokens de salida (Respuesta): {comp_t}")
        print(f"Tiempo total (TTFT + Generación): {duration:.2f}s")
        print(f"Eficiencia de ingesta: {prompt_t/duration:.2f} tokens_input/s")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Fallo en la conexión: {e}")
