import jinja2
import json
import requests

def debug_raw_injection():
    # 1. RENDERIZADO LOCAL (Igual que en tu Lab)
    with open("bunker_qwen_final.jinja", "r") as f:
        template_str = f.read()
    with open("debug_input.json", "r") as f:
        data = json.load(f)

    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.filters["tojson"] = lambda obj: json.dumps(obj, ensure_ascii=False)
    template = env.from_string(template_str)

    # Creamos el prompt exactamente como lo vería la GPU
    rendered_prompt = template.render(
        messages=data.get("messages", []),
        tools=data.get("tools", []),
        add_generation_prompt=True
    )

    # 2. INYECCIÓN DIRECTA AL ENDPOINT RAW
    # Usamos /completions (no chat) para enviar el texto final
    VLLM_URL = "http://localhost:8000/v1/completions"
    
    payload = {
        "model": "qwen2.5-coder-32b-awq",
        "prompt": rendered_prompt, # <--- Enviamos el texto YA traducido
        "max_tokens": 500,
        "temperature": 0,
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }

    print("--- 🧠 Enviando Prompt Renderizado a la GPU ---")
    response = requests.post(VLLM_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*50)
        print("🎬 RESPUESTA DIRECTA DE LA GPU:")
        print("="*50)
        print(result['choices'][0]['text'])
        print("="*50)
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    debug_raw_injection()