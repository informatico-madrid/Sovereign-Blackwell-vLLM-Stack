# Protocolo de Pensamiento Roo Code (v3.30+)

Para modelos que usan el provider 'OpenAI Compatible', el razonamiento debe integrarse en el campo `content` mediante etiquetas XML internas.

## Campos del JSON:
- **content**: 
    - `<thinking>...</thinking>`: Bloque de razonamiento interno (Chain of Thought).
    - [Texto Plano]: Mensaje dirigido al usuario.
- **tool_calls**: Bloque de ejecución técnica.

## Ejemplo de Respuesta Correcta:
{
  "role": "assistant",
  "content": "<thinking>Necesito crear un archivo markdown. Usaré write_to_file.</thinking> Hecho, aquí tienes el archivo.",
  "tool_calls": [...]
}
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "chatcmpl-tool-b40db47ed6766dd9",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": {
                    "path": "hola.md",
                    "content": "Soy Roo, un ingeniero de software altamente capacitado con amplios conocimientos en muchos lenguajes de programación, frameworks, patrones de diseño y mejores prácticas."
                }
            }
        }
    ],
    "function_call": null,
    "provider_specific_fields": null
}