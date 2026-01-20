#!/usr/bin/env python3
"""
Debug: Renderizar la plantilla Jinja para ver el prompt EXACTO
"""
from jinja2 import Template

# Leer la plantilla
with open('/mnt/bunker_data/ai/sovereign-ai-stack/bunker_qwen_final.jinja', 'r') as f:
    template_str = f.read()

template = Template(template_str)

# Datos de prueba
messages = [
    {"role": "user", "content": "crea un archivo llamado test.txt con el contenido 'hola mundo'"}
]

tools = [
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
]

# Renderizar
result = template.render(
    messages=messages,
    tools=tools,
    add_generation_prompt=True
)

print("="*60)
print("PROMPT RENDERIZADO:")
print("="*60)
print(result)
print("="*60)
