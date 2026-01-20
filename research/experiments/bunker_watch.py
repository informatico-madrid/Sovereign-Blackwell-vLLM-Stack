import time
import os
import subprocess

# Configuración
TEMPLATE_FILE = "bunker_qwen_final.jinja"
DEBUG_SCRIPT = "debug_template.py"

def get_mtime():
    return os.path.getmtime(TEMPLATE_FILE)

last_mtime = get_mtime()

print(f"--- 🚀 Bunker Lab: Observando cambios en {TEMPLATE_FILE} ---")

try:
    while True:
        current_mtime = get_mtime()
        if current_mtime != last_mtime:
            # Limpiar consola (funciona en Linux/SSH)
            os.system('clear')
            print(f"--- 🔄 Cambio detectado en {TEMPLATE_FILE}. Renderizando... ---")
            
            # Ejecutar el script de debug
            subprocess.run(["python3", DEBUG_SCRIPT])
            
            last_mtime = current_mtime
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDeteniendo Bunker Lab...")