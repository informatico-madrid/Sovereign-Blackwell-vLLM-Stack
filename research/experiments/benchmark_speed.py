import requests
import time

URL = "http://localhost:4000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer sk-master-bunker-2026"}
PAYLOAD = {
    "model": "bunker-agent",
    "messages": [{"role": "user", "content": "Write a 500-word essay about the future of Linux kernels."}],
    "stream": False
}

start = time.time()
response = requests.post(URL, headers=HEADERS, json=PAYLOAD).json()
end = time.time()

tokens = response['usage']['completion_tokens']
duration = end - start
print(f"--- BENCHMARK RESULT ---")
print(f"Tokens generados: {tokens}")
print(f"Tiempo total: {duration:.2f}s")
print(f"Throughput: {tokens/duration:.2f} tokens/s")
