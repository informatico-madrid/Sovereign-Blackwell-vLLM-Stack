# SPDX-License-Identifier: Apache-2.0
# Custom Tool Parser for Roo Code / Qwen2.5
# Based on vLLM Hermes2ProToolParser
# Version: 2.0 - Heavy Debug Version

"""
=============================================================================
ROO TOOL PARSER - MULTI-FORMAT TOOL CALL EXTRACTOR
=============================================================================

PROPÓSITO:
Este parser convierte las respuestas del modelo Qwen2.5 al formato OpenAI 
tool_calls que espera Roo Code.

FORMATOS SOPORTADOS (en orden de prioridad):
1. Hermes: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
2. Qwen confused: <tools>{"name": "...", "arguments": {...}}</tools>  
3. JSON directo: {"name": "...", "arguments": {...}}
4. XML Roo: <tool_name><param>value</param></tool_name>

REGLAS DE ROO CODE:
- content = "voz" del asistente (texto para el usuario)
- tool_calls = array de herramientas a ejecutar
- Cuando hay tool_calls, content puede ser null o tener texto ANTES del tool

PROBLEMA A RESOLVER:
El modelo genera XML como <write_to_file>...</write_to_file>
Debemos convertirlo a: tool_calls=[{name: "write_to_file", arguments: {...}}]
Y asegurar que content NO contenga fragmentos del XML
=============================================================================
"""

import json
from collections.abc import Sequence
from typing import Optional
import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================

# Lista de herramientas conocidas de Roo Code
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_separator(title: str):
    """Log a visual separator for better readability"""
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def log_debug_object(name: str, obj, max_len: int = 500):
    """Log an object with truncation for readability"""
    obj_str = str(obj)
    if len(obj_str) > max_len:
        obj_str = obj_str[:max_len] + "... [TRUNCATED]"
    logger.info(f"  {name}: {obj_str}")


def log_extraction_result(result: "ExtractedToolCallInformation"):
    """Log the final extraction result"""
    logger.info("=" * 80)
    logger.info("🟢 EXTRACTION RESULT")
    logger.info("=" * 80)
    logger.info(f"  tools_called: {result.tools_called}")
    logger.info(f"  content: {repr(result.content)}")
    if result.tool_calls:
        logger.info(f"  tool_calls ({len(result.tool_calls)}):")
        for i, tc in enumerate(result.tool_calls):
            logger.info(f"    [{i}] id={tc.id}, name={tc.function.name}")
            logger.info(f"        arguments={tc.function.arguments}")
    else:
        logger.info("  tool_calls: []")
    logger.info("=" * 80)
    return result


def make_extraction_result(tools_called: bool, tool_calls: list, content: Optional[str]) -> "ExtractedToolCallInformation":
    """Create and log extraction result"""
    result = ExtractedToolCallInformation(
        tools_called=tools_called,
        tool_calls=tool_calls,
        content=content
    )
    return log_extraction_result(result)


def extract_thinking_from_content(content: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extrae el "thinking/reasoning" del content.
    
    El modelo puede generar:
    - <think>...</think> seguido de tool call
    - Patrones de log con timestamps
    
    Args:
        content: El texto completo del modelo
        
    Returns:
        (thinking, clean_content) - El pensamiento extraído y el content limpio
    """
    if not content:
        return None, None
    
    import re as stdlib_re
    
    # Detectar <think>...</think> tags
    think_pattern = stdlib_re.compile(r'^<think>(.*?)</think>\s*', stdlib_re.DOTALL)
    match = think_pattern.match(content)
    if match:
        thinking = match.group(1).strip()
        clean_content = content[match.end():].strip()
        logger.info(f"[THINKING] Extracted <think> block: {thinking[:100]}...")
        return thinking, clean_content if clean_content else None
    
    # Detectar el patrón común de log con timestamp
    log_pattern = stdlib_re.compile(
        r'^(- \d{4}-\d{2}-\d{2}[^\n]*\n(?:\s*- (?:Result|Next|Step)[^\n]*\n)*)',
        stdlib_re.MULTILINE
    )
    
    match = log_pattern.match(content)
    if match:
        thinking = match.group(1).strip()
        clean_content = content[match.end():].strip()
        logger.info(f"[THINKING] Extracted log block: {thinking[:100]}...")
        return thinking, clean_content if clean_content else None
    
    # Si no hay patrón de pensamiento, devolver el content original
    return None, content


def clean_content_from_xml_tags(content: str, tool_name: str) -> Optional[str]:
    """
    Limpia el content removiendo cualquier fragmento de tag XML.
    
    Args:
        content: El texto a limpiar
        tool_name: Nombre de la herramienta para buscar sus tags
        
    Returns:
        Content limpio o None si está vacío
    """
    if not content:
        return None
    
    logger.info(f"[CLEAN] Input content: '{content[:100]}...'")
    
    # Remover cualquier tag parcial o completo de la herramienta
    # Patrones a remover:
    patterns_to_remove = [
        f"<{tool_name}>",      # Tag de apertura completo
        f"</{tool_name}>",     # Tag de cierre completo
        f"<{tool_name}",       # Tag de apertura parcial
        f"</{tool_name}",      # Tag de cierre parcial
    ]
    
    cleaned = content
    for pattern in patterns_to_remove:
        if pattern in cleaned:
            logger.info(f"[CLEAN] Removing pattern: '{pattern}'")
            cleaned = cleaned.replace(pattern, "")
    
    # También limpiar cualquier tag de herramienta Roo
    for roo_tool in ROO_TOOLS:
        patterns = [f"<{roo_tool}>", f"</{roo_tool}>", f"<{roo_tool}", f"</{roo_tool}"]
        for pattern in patterns:
            if pattern in cleaned:
                logger.info(f"[CLEAN] Removing Roo tool pattern: '{pattern}'")
                cleaned = cleaned.replace(pattern, "")
    
    # Limpiar tags Hermes también
    hermes_patterns = ["<tool_call>", "</tool_call>", "<tool_call", "</tool_call",
                       "<tools>", "</tools>", "<tools", "</tools"]
    for pattern in hermes_patterns:
        if pattern in cleaned:
            logger.info(f"[CLEAN] Removing Hermes pattern: '{pattern}'")
            cleaned = cleaned.replace(pattern, "")
    
    cleaned = cleaned.strip()
    
    logger.info(f"[CLEAN] Output content: '{cleaned[:100] if cleaned else 'NULL'}...'")
    
    return cleaned if cleaned else None


# =============================================================================
# MAIN PARSER CLASS
# =============================================================================

class RooToolParser(ToolParser):
    """
    Parser multi-formato para herramientas de Roo Code.
    
    Soporta:
    - Formato Hermes: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    - Formato Qwen: <tools>{"name": ..., "arguments": ...}</tools>
    - JSON directo: {"name": ..., "arguments": ...}
    - Formato XML Roo: <tool_name><param>value</param></tool_name>
    """
    
    def __init__(self, tokenizer: TokenizerLike):
        log_separator("RooToolParser.__init__")
        super().__init__(tokenizer)

        if isinstance(tokenizer, MistralTokenizer):
            logger.error("[INIT] ERROR: Detected Mistral tokenizer - not supported!")
            self.model_tokenizer = tokenizer.tokenizer
        
        logger.info(f"[INIT] Tokenizer type: {type(self.model_tokenizer)}")

        # Estado para streaming
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self.buffered_delta_text = ""

        # Tokens Hermes
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        # Regex patterns
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        
        tool_pattern = "|".join(ROO_TOOLS)
        self.roo_tool_regex = re.compile(
            rf"<({tool_pattern})>(.*?)</\1>|<({tool_pattern})>(.*)",
            re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError("Model tokenizer is required for RooToolParser")
            
        # Tokenize start/end tokens for streaming
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]
        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]
        
        logger.info(f"[INIT] ROO_TOOLS registered: {ROO_TOOLS}")
        logger.info(f"[INIT] tool_call_start_token_array: {self.tool_call_start_token_array}")
        logger.info("[INIT] RooToolParser initialized successfully")

    # =========================================================================
    # FORMAT DETECTION METHODS
    # =========================================================================

    def _has_hermes_format(self, text: str) -> bool:
        """Detecta formato Hermes: <tool_call>...</tool_call>"""
        result = self.tool_call_start_token in text
        logger.debug(f"[DETECT] _has_hermes_format: {result}")
        return result

    def _has_tools_tag_format(self, text: str) -> bool:
        """Detecta formato Qwen confundido: <tools>...</tools>"""
        result = "<tools>" in text and "name" in text and "arguments" in text
        logger.debug(f"[DETECT] _has_tools_tag_format: {result}")
        return result

    def _has_roo_xml_format(self, text: str) -> bool:
        """Detecta formato XML de Roo: <tool_name>...</tool_name>"""
        for tool_name in ROO_TOOLS:
            if f"<{tool_name}>" in text:
                logger.debug(f"[DETECT] _has_roo_xml_format: True (found <{tool_name}>)")
                return True
        logger.debug("[DETECT] _has_roo_xml_format: False")
        return False

    def _has_direct_json_format(self, text: str) -> bool:
        """
        Detecta JSON directo: {"name": ..., "arguments": ...}
        También detecta: <think>...</think>{"name": ..., "arguments": ...}
        """
        text = text.strip()
        
        # Caso 1: JSON directo al inicio
        if text.startswith('{') and '"name"' in text and '"arguments"' in text:
            logger.debug("[DETECT] _has_direct_json_format: True (starts with {)")
            return True
        
        # Caso 2: <think>...</think> seguido de JSON
        import re as stdlib_re
        think_pattern = stdlib_re.compile(r'^<think>.*?</think>\s*\{', stdlib_re.DOTALL)
        if think_pattern.match(text) and '"name"' in text and '"arguments"' in text:
            logger.debug("[DETECT] _has_direct_json_format: True (after </think>)")
            return True
        
        logger.debug("[DETECT] _has_direct_json_format: False")
        return False

    def _is_accumulating_xml_tag(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Verifica si el texto está acumulando un tag XML (parcial).
        
        IMPORTANTE: Detecta tanto tags completos sin cerrar como tags parciales.
        Ejemplos:
            "<write_to_file>..." sin </write_to_file> -> True
            "<write_to" (parcial) -> True  
            "<wri" (parcial) -> True
            "<" solo -> True (podría ser inicio de tag)
        
        Returns:
            (is_accumulating, tool_name_if_found)
        """
        # Primero verificar si termina con "<" que podría ser inicio de tag
        if text.rstrip().endswith("<"):
            logger.debug("[DETECT] Text ends with '<' - potential tag start")
            return True, None
        
        # Verificar tags parciales al final del texto
        # Buscar si hay un "<" seguido de caracteres que podrían ser nombre de tool
        last_lt = text.rfind("<")
        if last_lt >= 0:
            after_lt = text[last_lt:]
            # Si no hay ">" después del "<", podría ser tag parcial
            if ">" not in after_lt:
                # Verificar si coincide con inicio de alguna herramienta
                for tool_name in ROO_TOOLS + ["tool_call", "tools"]:
                    partial_tag = f"<{tool_name}"
                    # Verificar si after_lt es prefijo de partial_tag o al revés
                    if after_lt.startswith("</"):
                        # Es un tag de cierre parcial
                        close_partial = f"</{tool_name}"
                        if close_partial.startswith(after_lt) or after_lt.startswith(close_partial):
                            logger.debug(f"[DETECT] Partial closing tag detected: {after_lt}")
                            return True, tool_name
                    else:
                        # Es un tag de apertura parcial
                        if partial_tag.startswith(after_lt) or after_lt.startswith(partial_tag):
                            logger.debug(f"[DETECT] Partial opening tag detected: {after_lt} (matches {tool_name})")
                            return True, tool_name
        
        # Verificar tags completos sin cerrar
        for tool_name in ROO_TOOLS:
            full_start = f"<{tool_name}>"
            full_end = f"</{tool_name}>"
            
            if full_start in text and full_end not in text:
                logger.debug(f"[DETECT] Accumulating complete tag for: {tool_name}")
                return True, tool_name
        
        # También verificar tags Hermes y tools
        if "<tool_call>" in text and "</tool_call>" not in text:
            logger.debug("[DETECT] Accumulating <tool_call> tag")
            return True, "tool_call"
        if "<tools" in text and "</tools>" not in text:
            logger.debug("[DETECT] Accumulating <tools> tag")
            return True, "tools"
            
        return False, None

    # =========================================================================
    # REQUEST ADJUSTMENT
    # =========================================================================

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        log_separator("adjust_request")
        logger.info(f"[ADJUST] tools present: {bool(request.tools)}")
        logger.info(f"[ADJUST] tool_choice: {request.tool_choice}")
        
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            logger.info("[ADJUST] Set skip_special_tokens = False")
        
        return request

    # =========================================================================
    # MAIN EXTRACTION METHOD (NON-STREAMING)
    # =========================================================================

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        MÉTODO PRINCIPAL: Extrae tool calls del output completo del modelo.
        
        Este método es llamado cuando NO hay streaming (respuesta completa).
        """
        # =====================================================================
        # RAW DEBUG: Log completo del output del modelo
        # =====================================================================
        logger.info("=" * 80)
        logger.info("🔴 RAW MODEL OUTPUT (COMPLETE - NON-STREAMING)")
        logger.info("=" * 80)
        logger.info(f"Length: {len(model_output)} chars")
        logger.info("-" * 40)
        # Log completo sin truncar para debugging
        for i, line in enumerate(model_output.split('\n')):
            logger.info(f"LINE {i:03d}: {repr(line)}")
        logger.info("-" * 40)
        logger.info(f"RAW REPR: {repr(model_output)}")
        logger.info("=" * 80)
        
        log_separator("extract_tool_calls (NON-STREAMING)")
        logger.info(f"[EXTRACT] Input length: {len(model_output)} chars")
        log_debug_object("[EXTRACT] model_output", model_output, 500)
        
        # PASO 1: Detectar formato
        logger.info("[EXTRACT] Step 1: Detecting format...")
        
        has_hermes = self._has_hermes_format(model_output)
        has_tools = self._has_tools_tag_format(model_output)
        has_roo = self._has_roo_xml_format(model_output)
        has_json = self._has_direct_json_format(model_output)
        
        logger.info(f"[EXTRACT] Format detection results:")
        logger.info(f"  - Hermes (<tool_call>): {has_hermes}")
        logger.info(f"  - Qwen (<tools>): {has_tools}")
        logger.info(f"  - Roo XML (<tool_name>): {has_roo}")
        logger.info(f"  - Direct JSON: {has_json}")
        
        # PASO 2: Extraer según formato (orden de prioridad)
        logger.info("[EXTRACT] Step 2: Extracting based on format...")
        
        if has_hermes:
            logger.info("[EXTRACT] Using Hermes extractor")
            return self._extract_hermes_tool_calls(model_output, request)
        
        if has_tools:
            logger.info("[EXTRACT] Using <tools> tag extractor")
            return self._extract_tools_tag_format(model_output, request)
        
        if has_json:
            logger.info("[EXTRACT] Using direct JSON extractor")
            return self._extract_direct_json(model_output, request)
        
        if has_roo:
            logger.info("[EXTRACT] Using Roo XML extractor")
            return self._extract_roo_xml_tool_calls(model_output, request)
        
        # No se detectó ningún formato de tool call
        logger.info("[EXTRACT] No tool call format detected - returning as content")
        return make_extraction_result(
            tools_called=False, 
            tool_calls=[], 
            content=model_output
        )

    # =========================================================================
    # FORMAT-SPECIFIC EXTRACTORS
    # =========================================================================

    def _extract_hermes_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extrae tool calls en formato Hermes: <tool_call>JSON</tool_call>"""
        log_separator("_extract_hermes_tool_calls")
        
        try:
            # Buscar todos los matches
            function_call_tuples = self.tool_call_regex.findall(model_output)
            logger.info(f"[HERMES] Found {len(function_call_tuples)} tool_call matches")
            
            raw_function_calls = []
            for i, match in enumerate(function_call_tuples):
                json_str = match[0] if match[0] else match[1]
                logger.info(f"[HERMES] Match {i}: {json_str[:200]}...")
                try:
                    parsed = json.loads(json_str)
                    raw_function_calls.append(parsed)
                    logger.info(f"[HERMES] Parsed tool: {parsed.get('name', 'UNKNOWN')}")
                except json.JSONDecodeError as e:
                    logger.error(f"[HERMES] JSON parse error: {e}")
            
            # Construir ToolCall objects
            tool_calls = []
            for fc in raw_function_calls:
                tc = ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=fc["name"],
                        arguments=json.dumps(fc["arguments"], ensure_ascii=False),
                    ),
                )
                tool_calls.append(tc)
                logger.info(f"[HERMES] Created ToolCall: {fc['name']}")

            # Extraer content (texto ANTES del primer tool_call)
            content_end_idx = model_output.find(self.tool_call_start_token)
            raw_content = model_output[:content_end_idx] if content_end_idx > 0 else ""
            content = clean_content_from_xml_tags(raw_content, "tool_call")
            
            logger.info(f"[HERMES] Final content: {content[:100] if content else 'NULL'}...")
            logger.info(f"[HERMES] Total tool_calls: {len(tool_calls)}")
            
            return make_extraction_result(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception as e:
            logger.exception(f"[HERMES] ERROR: {e}")
            return make_extraction_result(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_tools_tag_format(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extrae tool calls cuando Qwen usa <tools> en lugar de <tool_call>"""
        log_separator("_extract_tools_tag_format")
        
        try:
            tools_regex = re.compile(r"<tools>(.*?)</tools>|<tools>(.*)", re.DOTALL)
            matches = tools_regex.findall(model_output)
            logger.info(f"[TOOLS_TAG] Found {len(matches)} <tools> matches")
            
            if not matches:
                logger.info("[TOOLS_TAG] No matches - returning as content")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            
            tool_calls = []
            for i, match in enumerate(matches):
                json_str = (match[0] if match[0] else match[1]).strip()
                logger.info(f"[TOOLS_TAG] Match {i}: {json_str[:200]}...")
                
                try:
                    function_call = json.loads(json_str)
                    tc = ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                        ),
                    )
                    tool_calls.append(tc)
                    logger.info(f"[TOOLS_TAG] Created ToolCall: {function_call['name']}")
                except json.JSONDecodeError as e:
                    logger.warning(f"[TOOLS_TAG] JSON parse error: {e}")
            
            if tool_calls:
                content_end_idx = model_output.find("<tools>")
                raw_content = model_output[:content_end_idx] if content_end_idx > 0 else ""
                content = clean_content_from_xml_tags(raw_content, "tools")
                
                logger.info(f"[TOOLS_TAG] Final content: {content[:100] if content else 'NULL'}...")
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content,
                )
            
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
            
        except Exception as e:
            logger.exception(f"[TOOLS_TAG] ERROR: {e}")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_direct_json(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extrae tool calls cuando el modelo responde con JSON directo o después de <think>"""
        log_separator("_extract_direct_json")
        
        try:
            # Verificar si hay <think> antes del JSON
            import re as stdlib_re
            think_pattern = stdlib_re.compile(r'^(<think>.*?</think>)\s*', stdlib_re.DOTALL)
            match = think_pattern.match(model_output)
            
            thinking_content = None
            json_part = model_output
            
            if match:
                thinking_content = match.group(1)  # Mantener tags <think></think> en content
                json_part = model_output[match.end():].strip()
                logger.info(f"[JSON] Found <think> block, extracting JSON after it")
                logger.info(f"[JSON] thinking_content: {thinking_content[:100]}...")
            
            json_str = json_part.strip()
            logger.info(f"[JSON] Parsing: {json_str[:200]}...")
            
            function_call = json.loads(json_str)
            
            if "name" in function_call and "arguments" in function_call:
                tc = ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                    ),
                )
                logger.info(f"[JSON] Created ToolCall: {function_call['name']}")
                
                # Si había thinking, ponerlo como content
                return make_extraction_result(
                    tools_called=True,
                    tool_calls=[tc],
                    content=thinking_content,  # <think>...</think> o None
                )
            
            logger.info("[JSON] JSON doesn't have name/arguments - returning as content")
            return make_extraction_result(
                tools_called=False, tool_calls=[], content=model_output
            )
            
        except Exception as e:
            logger.exception(f"[JSON] ERROR: {e}")
            return make_extraction_result(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_roo_xml_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extrae tool calls en formato XML de Roo.
        
        Formato esperado:
        <write_to_file>
        <path>archivo.txt</path>
        <content>contenido del archivo</content>
        </write_to_file>
        """
        log_separator("_extract_roo_xml_tool_calls")
        
        try:
            tool_calls = []
            content = None
            found_tool_name = None
            
            for tool_name in ROO_TOOLS:
                start_tag = f"<{tool_name}>"
                end_tag = f"</{tool_name}>"
                
                if start_tag not in model_output:
                    continue
                
                logger.info(f"[ROO_XML] Found tool: {tool_name}")
                found_tool_name = tool_name
                
                start_idx = model_output.find(start_tag)
                end_idx = model_output.find(end_tag)
                
                logger.info(f"[ROO_XML] start_idx={start_idx}, end_idx={end_idx}")
                
                if end_idx == -1:
                    # Tag no cerrado
                    xml_content = model_output[start_idx + len(start_tag):]
                    logger.warning(f"[ROO_XML] Tag not closed! Partial content: {xml_content[:200]}...")
                else:
                    xml_content = model_output[start_idx + len(start_tag):end_idx]
                
                logger.info(f"[ROO_XML] XML content: {xml_content[:300]}...")
                
                # Parsear parámetros XML
                arguments = {}
                param_pattern = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
                param_matches = param_pattern.findall(xml_content)
                
                logger.info(f"[ROO_XML] Found {len(param_matches)} parameters")
                
                for param_name, param_value in param_matches:
                    param_value = param_value.strip()
                    logger.info(f"[ROO_XML] Param '{param_name}': {param_value[:100]}...")
                    
                    # Intentar parsear JSON si aplica
                    if param_value.startswith('{') or param_value.startswith('['):
                        try:
                            param_value = json.loads(param_value)
                            logger.info(f"[ROO_XML] Param '{param_name}' parsed as JSON")
                        except json.JSONDecodeError:
                            pass
                    
                    arguments[param_name] = param_value
                
                # Crear ToolCall
                tc = ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=tool_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                tool_calls.append(tc)
                
                logger.info(f"[ROO_XML] Created ToolCall: {tool_name}")
                logger.info(f"[ROO_XML] Arguments: {arguments}")
                
                # Extraer content (texto ANTES del tool call)
                raw_content = model_output[:start_idx] if start_idx > 0 else ""
                content = clean_content_from_xml_tags(raw_content, tool_name)
                
                break  # Solo procesar una tool call a la vez
            
            if tool_calls:
                logger.info(f"[ROO_XML] SUCCESS - tool_calls: {len(tool_calls)}, content: {content[:50] if content else 'NULL'}")
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content,
                )
            
            logger.info("[ROO_XML] No tool calls extracted - returning as content")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        except Exception as e:
            logger.exception(f"[ROO_XML] ERROR: {e}")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # =========================================================================
    # STREAMING METHODS
    # =========================================================================

    def tool_call_delta_buffer(self, delta_text: str) -> str:
        """Buffer para manejar tokens parciales de <tool_call>"""
        if (
            delta_text in self.tool_call_start_token_array
            or delta_text in self.tool_call_end_token_array
        ):
            if (
                delta_text == self.tool_call_start_token_array[-1]
                or delta_text == self.tool_call_end_token_array[-1]
            ):
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        MÉTODO STREAMING: Extrae tool calls de forma incremental.
        
        CRÍTICO: 
        1. NO emitir content si hay tag XML parcial o completo
        2. NO emitir content si parece JSON de tool call (empieza con {)
        3. Debe actualizar prev_tool_call_arr y streamed_args_for_tool para vLLM
        4. El content acumulado por vLLM es la suma de TODOS los delta.content
           Por eso, si emitimos "{" como content, quedará ahí SIEMPRE
        """
        # =====================================================================
        # RAW DEBUG: Log cada chunk de streaming
        # =====================================================================
        # Log cada 50 chars o cuando hay tokens importantes
        should_log = (
            len(current_text) % 50 == 0 or 
            '<' in delta_text or 
            '{' in delta_text or
            '</think>' in current_text or
            '</tool_call>' in current_text
        )
        
        if should_log:
            logger.info("=" * 60)
            logger.info("🟡 STREAMING CHUNK")
            logger.info(f"  delta_text: {repr(delta_text)}")
            logger.info(f"  current_text ({len(current_text)} chars):")
            logger.info(f"  {repr(current_text[:500])}{'...' if len(current_text) > 500 else ''}")
            logger.info("=" * 60)
        
        # =====================================================================
        # PASO 0: Verificar si PODRÍA ser JSON directo (modelo ignoró template)
        # Si empieza con {, NO emitir NADA hasta saber qué es
        # =====================================================================
        stripped = current_text.strip()
        
        # =====================================================================
        # CASO ESPECIAL: <think>...</think> seguido de JSON
        # =====================================================================
        if stripped.startswith('<think>'):
            logger.debug("[STREAM] Detected <think> tag - checking for complete pattern")
            
            # Verificar si tenemos </think> Y JSON después
            if '</think>' in stripped:
                # Buscar JSON después de </think>
                import re as stdlib_re
                think_json_pattern = stdlib_re.compile(r'^<think>.*?</think>\s*(\{.*)', stdlib_re.DOTALL)
                match = think_json_pattern.match(stripped)
                
                if match:
                    json_part = match.group(1)
                    logger.debug(f"[STREAM] Found JSON after </think>: {json_part[:100]}...")
                    
                    # Verificar si el JSON está completo
                    if '"name"' in json_part and '"arguments"' in json_part:
                        brace_count = 0
                        json_complete = False
                        for char in json_part:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_complete = True
                                    break
                        
                        if json_complete:
                            logger.info("[STREAM] <think>+JSON complete - extracting")
                            return self._streaming_extract_think_json(current_text, delta_text, request)
                        else:
                            logger.debug("[STREAM] JSON after </think> incomplete - waiting")
                            return None
                    else:
                        logger.debug("[STREAM] JSON after </think> doesn't have name/arguments yet")
                        return None
                else:
                    # Tiene </think> pero no hay JSON todavía
                    logger.debug("[STREAM] Have </think> but no JSON yet - waiting")
                    return None
            else:
                # Aún no ha cerrado </think>
                logger.debug("[STREAM] <think> not closed yet - waiting")
                return None
        
        if stripped.startswith('{'):
            logger.debug(f"[STREAM] Text starts with '{{' - potential JSON tool call")
            
            # Si tiene "name", probablemente es tool call
            if '"name"' in stripped:
                logger.info("[STREAM] Detected JSON with 'name' field - likely tool call")
                
                # Verificar si el JSON está completo (tiene cierre de llaves balanceado)
                brace_count = 0
                json_complete = False
                for char in stripped:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_complete = True
                            break
                
                if json_complete:
                    logger.info("[STREAM] JSON complete - extracting tool call")
                    return self._streaming_extract_direct_json(current_text, delta_text, request)
                else:
                    logger.debug("[STREAM] JSON incomplete - waiting for more tokens")
                    return None  # NO emitir content, esperar JSON completo
            else:
                # Empieza con { pero aún no tiene "name"
                # Podría ser JSON de tool o algo else
                # Para ser seguros, NO emitir content
                logger.debug("[STREAM] Starts with '{' but no 'name' yet - suppressing content")
                return None  # NO emitir, esperar más tokens
        
        # =====================================================================
        # PASO 1: Verificar si current_text contiene CUALQUIER indicio de tool XML
        # Si es así, NO emitir NADA como content
        # =====================================================================
        
        # Verificar si hay "<" en el texto (podría ser inicio de tool)
        has_potential_tag = "<" in current_text
        
        if has_potential_tag:
            logger.debug(f"[STREAM] Text contains '<' - checking for tool patterns")
            
            # Verificar si es un tag XML completo de Roo
            if self._has_roo_xml_format(current_text):
                logger.info("[STREAM] Detected complete Roo XML format")
                return self._streaming_extract_roo_xml(current_text, delta_text, request)
            
            # Verificar si es formato Hermes completo
            if self._has_hermes_format(current_text) and "</tool_call>" in current_text:
                logger.info("[STREAM] Detected complete Hermes format")
                return self._streaming_extract_hermes(previous_text, current_text, delta_text, request)
            
            # Verificar si es formato <tools> completo
            if self._has_tools_tag_format(current_text) and "</tools>" in current_text:
                logger.info("[STREAM] Detected complete <tools> format")
                return self._streaming_extract_tools_format(current_text, delta_text, request)
            
            # Si llegamos aquí, hay un "<" pero no es un formato completo
            # Verificar si estamos acumulando un tag
            is_accumulating, acc_tool = self._is_accumulating_xml_tag(current_text)
            
            if is_accumulating:
                logger.debug(f"[STREAM] Accumulating tag for: {acc_tool} - suppressing content")
                return None  # NO emitir nada, esperar más tokens
            
            # Si hay "<" pero no coincide con ningún patrón conocido,
            # podría ser texto normal con "<" (ej: "a < b")
            # Verificar si el "<" está seguido de una letra (posible tag)
            import re as stdlib_re
            if stdlib_re.search(r'<[a-zA-Z_]', current_text):
                logger.debug("[STREAM] Found '<' followed by letter - likely tag, suppressing")
                return None
        
        # =====================================================================
        # PASO 2: No hay indicios de tool call - emitir como content normal
        # =====================================================================
        logger.debug(f"[STREAM] No tool pattern detected - emitting as content")
        return DeltaMessage(content=delta_text)

    def _streaming_extract_tools_format(
        self,
        current_text: str,
        delta_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Streaming extraction para formato <tools>"""
        logger.debug("[STREAM_TOOLS] Processing <tools> format")
        
        tools_regex = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
        match = tools_regex.search(current_text)
        
        if not match:
            logger.debug("[STREAM_TOOLS] No complete match yet")
            return None
        
        try:
            json_str = match.group(1).strip()
            function_call = json.loads(json_str)
            
            # Solo emitir una vez
            if self.current_tool_id < 0 or len(self.prev_tool_call_arr) == 0:
                self.current_tool_id += 1
                
                # CRÍTICO: Actualizar arrays para vLLM
                tool_call_dict = {
                    "name": function_call["name"],
                    "arguments": function_call.get("arguments", {})
                }
                self.prev_tool_call_arr.append(tool_call_dict)
                
                args_str = json.dumps(function_call.get("arguments", {}), ensure_ascii=False)
                self.streamed_args_for_tool.append(args_str)
                
                logger.info(f"[STREAM_TOOLS] Emitting tool call: {function_call['name']}")
                
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=function_call["name"],
                                arguments=args_str
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
        except json.JSONDecodeError as e:
            logger.debug(f"[STREAM_TOOLS] JSON incomplete: {e}")
        
        return None

    def _streaming_extract_roo_xml(
        self,
        current_text: str,
        delta_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Streaming extraction para formato XML de Roo"""
        logger.debug("[STREAM_ROO] Processing Roo XML format")
        
        for tool_name in ROO_TOOLS:
            start_tag = f"<{tool_name}>"
            end_tag = f"</{tool_name}>"
            
            if start_tag not in current_text or end_tag not in current_text:
                continue
            
            logger.debug(f"[STREAM_ROO] Found complete tags for: {tool_name}")
            
            # Verificar que hay contenido interno completo
            start_idx = current_text.find(start_tag)
            end_idx = current_text.find(end_tag)
            xml_content = current_text[start_idx + len(start_tag):end_idx]
            
            inner_tag_pattern = re.compile(r'<(\w+)>.*?</\1>', re.DOTALL)
            inner_matches = inner_tag_pattern.findall(xml_content)
            
            if len(inner_matches) == 0:
                logger.debug(f"[STREAM_ROO] No inner tags complete yet for {tool_name}")
                return None
            
            # Extraer usando el método principal
            result = self._extract_roo_xml_tool_calls(current_text, request)
            
            if not result.tools_called or not result.tool_calls:
                logger.debug("[STREAM_ROO] Extraction returned no tools")
                return None
            
            tc = result.tool_calls[0]
            args_dict = json.loads(tc.function.arguments) if tc.function.arguments != '{}' else {}
            
            if not args_dict:
                logger.warning(f"[STREAM_ROO] No arguments found for {tool_name}")
                return None
            
            # Solo emitir una vez
            if self.current_tool_id < 0 or len(self.prev_tool_call_arr) == 0:
                self.current_tool_id += 1
                
                # CRÍTICO: Actualizar arrays para vLLM
                tool_call_dict = {
                    "name": tc.function.name,
                    "arguments": args_dict
                }
                self.prev_tool_call_arr.append(tool_call_dict)
                self.streamed_args_for_tool.append(tc.function.arguments)
                
                logger.info(f"[STREAM_ROO] Emitting tool call: {tc.function.name}")
                logger.info(f"[STREAM_ROO] Arguments: {args_dict}")
                
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
        
        return None

    def _streaming_extract_hermes(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Streaming extraction para formato Hermes"""
        logger.debug("[STREAM_HERMES] Processing Hermes format")
        
        delta_text = self.tool_call_delta_buffer(delta_text)
        
        if (
            len(previous_text) >= len(self.buffered_delta_text)
            and previous_text[-len(self.buffered_delta_text):] == self.buffered_delta_text
        ):
            previous_text = previous_text[:-len(self.buffered_delta_text)]
            current_text = previous_text + delta_text

        if "</tool_call>" not in current_text:
            logger.debug("[STREAM_HERMES] No complete </tool_call> yet")
            return None
        
        result = self._extract_hermes_tool_calls(current_text, request)
        
        if not result.tools_called or not result.tool_calls:
            return None
        
        tc = result.tool_calls[0]
        
        if self.current_tool_id < 0 or len(self.prev_tool_call_arr) == 0:
            self.current_tool_id += 1
            
            args_dict = json.loads(tc.function.arguments)
            tool_call_dict = {
                "name": tc.function.name,
                "arguments": args_dict
            }
            self.prev_tool_call_arr.append(tool_call_dict)
            self.streamed_args_for_tool.append(tc.function.arguments)
            
            logger.info(f"[STREAM_HERMES] Emitting tool call: {tc.function.name}")
            
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        ).model_dump(exclude_none=True),
                    )
                ]
            )
        
        return None

    def _streaming_extract_think_json(
        self,
        current_text: str,
        delta_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Streaming extraction para formato <think>...</think> seguido de JSON.
        
        El modelo genera:
        <think>razonamiento interno</think>
        {"name": "tool_name", "arguments": {...}}
        
        Debemos:
        1. Extraer el <think>...</think> como content
        2. Extraer el JSON como tool_call
        """
        log_separator("_streaming_extract_think_json")
        
        import re as stdlib_re
        
        try:
            # Separar thinking del JSON
            think_pattern = stdlib_re.compile(r'^(<think>.*?</think>)\s*(\{.*)', stdlib_re.DOTALL)
            match = think_pattern.match(current_text.strip())
            
            if not match:
                logger.warning("[STREAM_THINK_JSON] Could not parse think+json pattern")
                return None
            
            thinking_content = match.group(1)  # <think>...</think>
            json_str = match.group(2)          # {"name": ..., "arguments": ...}
            
            logger.info(f"[STREAM_THINK_JSON] thinking_content: {thinking_content[:100]}...")
            logger.info(f"[STREAM_THINK_JSON] json_str: {json_str[:200]}...")
            
            function_call = json.loads(json_str)
            
            if "name" not in function_call:
                logger.warning("[STREAM_THINK_JSON] JSON missing 'name' field")
                return None
            
            tool_name = function_call["name"]
            arguments = function_call.get("arguments", {})
            
            logger.info(f"[STREAM_THINK_JSON] Tool: {tool_name}")
            
            # Solo emitir una vez
            if self.current_tool_id < 0 or len(self.prev_tool_call_arr) == 0:
                self.current_tool_id += 1
                
                args_str = json.dumps(arguments, ensure_ascii=False)
                
                # CRÍTICO: Actualizar arrays para vLLM
                tool_call_dict = {
                    "name": tool_name,
                    "arguments": arguments
                }
                self.prev_tool_call_arr.append(tool_call_dict)
                self.streamed_args_for_tool.append(args_str)
                
                logger.info(f"[STREAM_THINK_JSON] Emitting content + tool_call")
                logger.info(f"[STREAM_THINK_JSON] Content: {thinking_content[:100]}...")
                
                # Emitir el content (thinking) Y el tool_call juntos
                return DeltaMessage(
                    content=thinking_content,
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=tool_name,
                                arguments=args_str
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[STREAM_THINK_JSON] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[STREAM_THINK_JSON] Error: {e}")
            return None

    def _streaming_extract_direct_json(
        self,
        current_text: str,
        delta_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Streaming extraction para JSON directo.
        
        Esto ocurre cuando el modelo ignora el template y devuelve:
        {"name": "tool_name", "arguments": {...}}
        
        directamente sin ningún wrapper XML.
        """
        log_separator("_streaming_extract_direct_json")
        
        try:
            json_str = current_text.strip()
            logger.info(f"[STREAM_JSON] Parsing: {json_str[:300]}...")
            
            function_call = json.loads(json_str)
            
            # Verificar que tiene la estructura esperada
            if "name" not in function_call:
                logger.warning("[STREAM_JSON] JSON missing 'name' field")
                return None
            
            tool_name = function_call["name"]
            arguments = function_call.get("arguments", {})
            
            logger.info(f"[STREAM_JSON] Detected tool: {tool_name}")
            logger.info(f"[STREAM_JSON] Arguments: {arguments}")
            
            # Verificar que es una herramienta conocida
            if tool_name not in ROO_TOOLS:
                logger.warning(f"[STREAM_JSON] Unknown tool: {tool_name}")
                # Aún así procesarlo, podría ser válido
            
            # Solo emitir una vez
            if self.current_tool_id < 0 or len(self.prev_tool_call_arr) == 0:
                self.current_tool_id += 1
                
                args_str = json.dumps(arguments, ensure_ascii=False)
                
                # CRÍTICO: Actualizar arrays para vLLM
                tool_call_dict = {
                    "name": tool_name,
                    "arguments": arguments
                }
                self.prev_tool_call_arr.append(tool_call_dict)
                self.streamed_args_for_tool.append(args_str)
                
                logger.info(f"[STREAM_JSON] Emitting tool call: {tool_name}")
                
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=tool_name,
                                arguments=args_str
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
            else:
                logger.debug("[STREAM_JSON] Tool already emitted, skipping")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"[STREAM_JSON] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[STREAM_JSON] ERROR: {e}")
            return None
