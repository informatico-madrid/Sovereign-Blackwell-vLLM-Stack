#!/usr/bin/env python3
"""
Debug script para simular lo que hace el Hermes parser
"""
import json
import re

# Esto es lo que el parser espera
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

# Respuesta actual del modelo (formato XML de Roo)
model_output_bad = '''<write_to_file>
<path>hola.md</path>
<content>Soy Roo, un ingeniero de software altamente capacitado con conocimientos extensos en múltiples lenguajes de programación, frameworks, patrones de diseño y mejores prácticas.</content>
</write_to_file>'''

# Lo que el parser espera (formato Hermes)
model_output_good = '''<tool_call>
{"name": "write_to_file", "arguments": {"path": "hola.md", "content": "Soy Roo, un ingeniero de software altamente capacitado..."}}
</tool_call>'''

def extract_tool_calls(model_output: str):
    """Simula exactamente lo que hace HermesToolParser.extract_tool_calls"""
    
    print(f"\n{'='*60}")
    print(f"Checking: {model_output[:100]}...")
    print(f"{'='*60}")
    
    # sanity check; avoid unnecessary processing
    if TOOL_CALL_START not in model_output:
        print(f"❌ No encontró '{TOOL_CALL_START}' en el output")
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output
        }

    print(f"✅ Encontró '{TOOL_CALL_START}' en el output")
    
    try:
        # Buscar matches con regex
        function_call_tuples = TOOL_CALL_REGEX.findall(model_output)
        print(f"Regex matches: {function_call_tuples}")
        
        # Parsear JSON
        raw_function_calls = [
            json.loads(match[0] if match[0] else match[1])
            for match in function_call_tuples
        ]
        print(f"Parsed function calls: {raw_function_calls}")
        
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc["arguments"], ensure_ascii=False)
                }
            }
            for fc in raw_function_calls
        ]
        
        content = model_output[:model_output.find(TOOL_CALL_START)]
        
        return {
            "tools_called": True,
            "tool_calls": tool_calls,
            "content": content if content else None
        }
        
    except Exception as e:
        print(f"❌ Error parsing: {e}")
        return {
            "tools_called": False,
            "tool_calls": [],
            "content": model_output
        }


print("\n" + "="*60)
print("PROBANDO CON OUTPUT MALO (formato XML de Roo)")
print("="*60)
result = extract_tool_calls(model_output_bad)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")

print("\n" + "="*60)
print("PROBANDO CON OUTPUT BUENO (formato Hermes)")
print("="*60)
result = extract_tool_calls(model_output_good)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")

print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)
print("""
El problema es claro: 

El modelo está generando formato XML de Roo:
    <write_to_file>
    <path>hola.md</path>
    ...
    </write_to_file>

Pero el parser Hermes espera:
    <tool_call>
    {"name": "write_to_file", "arguments": {...}}
    </tool_call>

SOLUCIÓN: El modelo debe usar <tool_call> tags con JSON adentro,
NO tags XML con el nombre de la herramienta.
""")
