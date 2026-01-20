import jinja2
import json
import os

def render_bunker_template():
    # 1. Cargar el template
    template_path = "bunker_qwen_final.jinja"
    with open(template_path, "r") as f:
        template_str = f.read()

    # 2. Cargar el input (Copia el JSON de 'Input' de Langfuse a un archivo)
    # Si no existe, creamos uno de prueba básico
    input_path = "debug_input.json"
    if not os.path.exists(input_path):
        sample_input = {
            "messages": [
                {"role": "system", "content": "Eres un ingeniero."},
                {"role": "user", "content": "Crea el archivo test.py"}
            ],
            "tools": [{"name": "write_to_file", "description": "Crea archivos"}]
        }
        with open(input_path, "w") as f:
            json.dump(sample_input, f, indent=2)

    with open(input_path, "r") as f:
        data = json.load(f)

    # 3. Configurar Jinja2 para que se comporte como vLLM/HuggingFace
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    # vLLM añade este filtro para convertir objetos a JSON
    env.filters["tojson"] = lambda obj: json.dumps(obj, ensure_ascii=False)
    
    template = env.from_string(template_str)

    # 4. Renderizar
    rendered = template.render(
        messages=data.get("messages", []),
        tools=data.get("tools", []),
        add_generation_prompt=True
    )

    print("\n" + "="*50)
    print("PROMPT RENDERIZADO (Lo que ve la GPU):")
    print("="*50 + "\n")
    print(rendered)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    render_bunker_template()
