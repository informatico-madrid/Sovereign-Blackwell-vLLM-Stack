#!/usr/bin/env python3
"""
Debug: Ver el prompt EXACTO que se envía al modelo
"""
import requests
import json

# Simular una petición simple con tools
payload = {
    "model": "qwen2.5-coder-32b-awq",
    "messages": [
        {"role": "user", "content": "crea un archivo llamado test.txt con el contenido 'hola mundo'"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "write_to_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ],
    "max_tokens": 500,
    "temperature": 0.1
}

print("="*60)
print("Enviando petición a vLLM...")
print("="*60)

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload,
    headers={"Content-Type": "application/json"}
)

result = response.json()
print("\n" + "="*60)
print("RESPUESTA COMPLETA:")
print("="*60)
print(json.dumps(result, indent=2, ensure_ascii=False))

if "choices" in result:
    choice = result["choices"][0]
    message = choice.get("message", {})
    
    print("\n" + "="*60)
    print("ANÁLISIS:")
    print("="*60)
    print(f"Content: {message.get('content')}")
    print(f"Tool Calls: {message.get('tool_calls')}")
    
    if message.get('content') and not message.get('tool_calls'):
        print("\n⚠️  El modelo puso la respuesta en 'content' en lugar de 'tool_calls'")
        if '<tool_call>' in message.get('content', ''):
            print("   → Pero SÍ usó el tag <tool_call>! El parser no lo detectó.")
        elif '<write_to_file>' in message.get('content', ''):
            print("   → El modelo usó <write_to_file> en lugar de <tool_call>")
        else:
            print("   → El modelo no usó ningún formato de tool call")
