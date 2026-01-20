import re
import json

def simulate_vllm_parser(model_output):
    print("\n" + "="*50)
    print("SIMULADOR DE PARSER vLLM (qwen3_coder)")
    print("="*50)
    print(f"TEXTO RECIBIDO DE LA GPU:\n{model_output}\n")

    # El parser de vLLM busca exactamente este patrón:
    # <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        print("❌ ERROR: El parser no encontró etiquetas <tool_call> válidas.")
        print("Sugerencia: Revisa si el modelo olvidó las etiquetas o los saltos de línea.")
        return

    print(f"✅ SE ENCONTRARON {len(matches)} LLAMADAS A HERRAMIENTAS:")
    
    for i, match in enumerate(matches):
        try:
            tool_data = json.loads(match.strip())
            print(f"\nHerramienta {i+1}:")
            print(f"  - Nombre: {tool_data.get('name')}")
            print(f"  - Argumentos: {json.dumps(tool_data.get('arguments'), indent=4, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"  - ❌ ERROR: El contenido dentro de <tool_call> no es un JSON válido.")
            print(f"    Contenido problemático: {match}")

# --- PRUEBA EN VIVO ---
# Aquí pegas lo que veas en el 'content' del Output de Langfuse
output_del_modelo = """
Voy a crear el archivo solicitado.
<tool_call>
{"name": "write_to_file", "arguments": {"path": "bunker_status.md", "content": "| IP | Desc |\\n|---|---|\\n| 192.168.1.201 | Bunker |"}}
</tool_call>
"""

simulate_vllm_parser(output_del_modelo)