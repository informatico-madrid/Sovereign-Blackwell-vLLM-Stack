#!/usr/bin/env python3
"""
Test del RooToolParser
"""
import sys
sys.path.insert(0, '/mnt/bunker_data/ai/vllm/vllm-src')

import json

# Simular las clases necesarias
class FunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class ToolCall:
    def __init__(self, type, function):
        self.type = type
        self.function = function

class ExtractedToolCallInformation:
    def __init__(self, tools_called, tool_calls, content):
        self.tools_called = tools_called
        self.tool_calls = tool_calls
        self.content = content

# Importar regex
import re

# Lista de herramientas de Roo
ROO_TOOLS = [
    "execute_command",
    "read_file", 
    "write_to_file",
    "apply_diff",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "browser_action",
    "use_mcp_tool",
    "access_mcp_resource",
    "ask_followup_question",
    "attempt_completion",
    "switch_mode",
    "new_task",
    "fetch_instructions",
]


def extract_roo_xml_tool_calls(model_output: str):
    """Extrae tool calls en formato XML de Roo"""
    tool_calls = []
    content = model_output
    
    for tool_name in ROO_TOOLS:
        start_tag = f"<{tool_name}>"
        end_tag = f"</{tool_name}>"
        
        if start_tag in model_output:
            start_idx = model_output.find(start_tag)
            end_idx = model_output.find(end_tag)
            
            if end_idx == -1:
                xml_content = model_output[start_idx + len(start_tag):]
            else:
                xml_content = model_output[start_idx + len(start_tag):end_idx]
            
            # Parse XML parameters
            arguments = {}
            param_pattern = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
            matches = param_pattern.findall(xml_content)
            
            for param_name, param_value in matches:
                param_value = param_value.strip()
                if param_value.startswith('{') or param_value.startswith('['):
                    try:
                        param_value = json.loads(param_value)
                    except json.JSONDecodeError:
                        pass
                arguments[param_name] = param_value
            
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            })
            
            content = model_output[:start_idx].strip()
            print(f"✅ Extracted Roo XML tool call: {tool_name}")
            print(f"   Arguments: {json.dumps(arguments, indent=2, ensure_ascii=False)}")
            break
    
    return {
        "tools_called": len(tool_calls) > 0,
        "tool_calls": tool_calls,
        "content": content if content else None
    }


# Test cases
print("="*60)
print("TEST 1: Formato XML de Roo - write_to_file")
print("="*60)

test1 = '''<write_to_file>
<path>hola.md</path>
<content>Soy Roo, un ingeniero de software altamente capacitado con conocimientos extensos en múltiples lenguajes de programación, frameworks, patrones de diseño y mejores prácticas.</content>
</write_to_file>'''

result = extract_roo_xml_tool_calls(test1)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")


print("\n" + "="*60)
print("TEST 2: Formato XML de Roo - execute_command")
print("="*60)

test2 = '''<execute_command>
<command>ls -la</command>
<requires_approval>false</requires_approval>
</execute_command>'''

result = extract_roo_xml_tool_calls(test2)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")


print("\n" + "="*60)
print("TEST 3: Formato XML de Roo - read_file")
print("="*60)

test3 = '''<read_file>
<path>/etc/passwd</path>
</read_file>'''

result = extract_roo_xml_tool_calls(test3)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")


print("\n" + "="*60)
print("TEST 4: Con texto antes del tool call")
print("="*60)

test4 = '''Voy a crear el archivo que me pediste.

<write_to_file>
<path>test.txt</path>
<content>Hello World</content>
</write_to_file>'''

result = extract_roo_xml_tool_calls(test4)
print(f"\nResultado: {json.dumps(result, indent=2, ensure_ascii=False)}")


print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)
print("""
El parser RooToolParser puede convertir:

ENTRADA (formato XML de Roo):
<write_to_file>
<path>hola.md</path>
<content>...</content>
</write_to_file>

SALIDA (formato OpenAI tool_calls):
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "write_to_file",
      "arguments": "{\\"path\\": \\"hola.md\\", \\"content\\": \\"...\\"}"
    }
  }]
}
""")
