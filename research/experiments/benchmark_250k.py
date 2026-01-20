import requests
import time
import json

URL = "http://localhost:4000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer sk-master-bunker-2026", "Content-Type": "application/json"}

# Mantenemos los 250k tokens
base_entry = "TIMESTAMP: 2026-01-20 | MSG: PCIe_Gen5_Active\n"
massive_payload = base_entry * 5500 

payload = {
    "model": "bunker-agent",
    "messages": [{"role": "user", "content": massive_payload}],
    "stream": True, # <--- ACTIVAMOS STREAMING
    "max_tokens": 10
}

print("🚀 Medición de TTFT (Time To First Token) para 250K tokens...")
start_time = time.time()
response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), stream=True)

ttft = None
for line in response.iter_lines():
    if line:
        ttft = time.time() - start_time
        print(f"✅ ¡Primer token recibido!")
        break

print(f"--- RESULTADO ---")
print(f"TTFT: {ttft:.2f} segundos para ingerir 250,000 tokens.")
