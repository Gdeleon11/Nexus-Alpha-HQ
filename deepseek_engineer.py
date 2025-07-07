
#!/usr/bin/env python3
"""
DeepseekEngineer Module for DataNexus - Autonomous System Repair
Este módulo proporciona capacidades de auto-reparación del sistema usando IA.
"""

import os
import sys
import json
import sqlite3
import asyncio
import subprocess
import traceback
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

class DeepseekEngineer:
    """
    Ingeniero de Software AI residente usando Deepseek para auto-reparación y generación de herramientas.
    """

    def __init__(self, socketio=None):
        """
        Inicializar el Ingeniero Deepseek.

        Args:
            socketio: Instancia de SocketIO para comunicación con el dashboard
        """
        self.socketio = socketio

        # Configurar cliente Deepseek usando OpenAI SDK
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if self.deepseek_api_key:
            try:
                self.client = OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                self._log("✅ Deepseek Engineer inicializado correctamente")
            except Exception as e:
                self._log(f"❌ Error configurando Deepseek: {e}")
                self.client = None
        else:
            self._log("⚠️ DEEPSEEK_API_KEY no encontrada. Ingeniero funcionará en modo simulado.")
            self.client = None

        # Historial de reparaciones
        self.repair_history = []

        # Herramientas generadas dinámicamente
        self.generated_tools = {}

    def _log(self, message: str):
        """Log a message to dashboard and console."""
        print(f"🤖 DeepseekEngineer: {message}")
        if self.socketio:
            try:
                self.socketio.emit('new_log', {'data': f"🤖 DeepseekEngineer: {message}"})
            except Exception as e:
                print(f"Error emitiendo log: {e}")

    async def self_heal_on_error(self, error_traceback: str, file_path: str, 
                                code_snippet: str = None) -> Dict[str, Any]:
        """
        Auto-reparación de errores de código usando Deepseek.

        Args:
            error_traceback: Traceback completo del error
            file_path: Ruta del archivo que causó el error
            code_snippet: Fragmento de código específico (opcional)

        Returns:
            Diccionario con la propuesta de reparación
        """
        try:
            self._log(f"🔧 Iniciando auto-reparación para error en {file_path}")

            # Leer el archivo completo si no se proporciona snippet
            if not code_snippet and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_snippet = f.read()

            if not code_snippet:
                return {
                    'success': False,
                    'error': 'No se pudo obtener el código fuente'
                }

            # Extraer información del error
            error_info = self._parse_error_traceback(error_traceback)

            # Crear prompt para Deepseek
            repair_prompt = self._create_repair_prompt(
                error_traceback, file_path, code_snippet, error_info
            )

            # Llamar a Deepseek para obtener la solución
            if self.client:
                response = await self._call_deepseek_api(repair_prompt)

                if response:
                    # Procesar la respuesta de Deepseek
                    repair_result = self._process_repair_response(
                        response, file_path, error_info
                    )

                    # Guardar en historial
                    self.repair_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'file_path': file_path,
                        'error_info': error_info,
                        'repair_result': repair_result,
                        'original_error': error_traceback
                    })

                    self._log(f"✅ Propuesta de reparación generada para {file_path}")
                    return repair_result
                else:
                    return {
                        'success': False,
                        'error': 'No se pudo obtener respuesta de Deepseek'
                    }
            else:
                # Modo simulado
                return self._simulate_repair(error_info, file_path)

        except Exception as e:
            self._log(f"❌ Error en auto-reparación: {e}")
            return {
                'success': False,
                'error': f'Error en proceso de auto-reparación: {str(e)}'
            }

    async def create_new_analyzer(self, tool_description: str) -> Dict[str, Any]:
        """
        Generar una nueva herramienta de análisis usando Deepseek.

        Args:
            tool_description: Descripción de la herramienta solicitada

        Returns:
            Diccionario con la nueva herramienta generada
        """
        try:
            self._log(f"🛠️ Generando nueva herramienta: {tool_description}")

            # Crear prompt para generación de herramientas
            tool_prompt = self._create_tool_generation_prompt(tool_description)

            if self.client:
                response = await self._call_deepseek_api(tool_prompt)

                if response:
                    # Procesar la respuesta y generar el código
                    tool_result = self._process_tool_response(response, tool_description)

                    # Guardar herramienta generada
                    tool_id = f"tool_{len(self.generated_tools) + 1}"
                    self.generated_tools[tool_id] = {
                        'description': tool_description,
                        'code': tool_result.get('code', ''),
                        'created_at': datetime.now().isoformat(),
                        'function_name': tool_result.get('function_name', 'new_analyzer')
                    }

                    self._log(f"✅ Nueva herramienta '{tool_id}' generada exitosamente")
                    return {
                        'success': True,
                        'tool_id': tool_id,
                        'tool_data': self.generated_tools[tool_id],
                        'ready_for_approval': True
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No se pudo generar la herramienta'
                    }
            else:
                # Modo simulado
                return self._simulate_tool_generation(tool_description)

        except Exception as e:
            self._log(f"❌ Error generando herramienta: {e}")
            return {
                'success': False,
                'error': f'Error en generación de herramienta: {str(e)}'
            }

    def _parse_error_traceback(self, error_traceback: str) -> Dict[str, Any]:
        """Parsear el traceback para extraer información útil."""
        error_info = {
            'error_type': 'Unknown',
            'error_message': '',
            'line_number': None,
            'function_name': None,
            'file_name': None
        }

        try:
            lines = error_traceback.split('\n')

            # Buscar el tipo de error y mensaje
            for line in reversed(lines):
                if ':' in line and any(err in line for err in ['Error', 'Exception']):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        error_info['error_type'] = parts[0].strip()
                        error_info['error_message'] = parts[1].strip()
                        break

            # Buscar línea del error
            for line in lines:
                if 'line' in line.lower() and ',' in line:
                    match = re.search(r'line (\d+)', line)
                    if match:
                        error_info['line_number'] = int(match.group(1))
                        break

            # Buscar archivo y función
            for line in lines:
                if 'File' in line and '.py' in line:
                    match = re.search(r'File "([^"]+)"', line)
                    if match:
                        error_info['file_name'] = match.group(1)
                        break

        except Exception as e:
            print(f"Error parseando traceback: {e}")

        return error_info

    def _create_repair_prompt(self, error_traceback: str, file_path: str, 
                            code_snippet: str, error_info: Dict[str, Any]) -> str:
        """Crear prompt para reparación de código."""
        return f"""Eres un ingeniero de software senior experto en Python. El siguiente código ha producido este error. Analiza la causa raíz y reescribe únicamente el fragmento de código necesario para corregirlo de forma permanente.

INFORMACIÓN DEL ERROR:
- Tipo: {error_info['error_type']}
- Mensaje: {error_info['error_message']}
- Línea: {error_info['line_number']}
- Archivo: {file_path}

TRACEBACK COMPLETO:
{error_traceback}

CÓDIGO FUENTE:
```python
{code_snippet}
```

INSTRUCCIONES:
1. Identifica la causa raíz del error
2. Proporciona ÚNICAMENTE el fragmento de código corregido
3. Incluye comentarios explicando la corrección
4. Asegúrate de que la corrección sea robusta y no cause otros errores

RESPUESTA ESPERADA:
```python
# Código corregido aquí
```

EXPLICACIÓN: [Breve explicación de la causa y la corrección]"""

    def _create_tool_generation_prompt(self, tool_description: str) -> str:
        """Crear prompt para generación de herramientas."""
        return f"""Eres un ingeniero de software experto en análisis financiero y trading. Necesito que generes una función Python que implemente la siguiente funcionalidad:

DESCRIPCIÓN DE LA HERRAMIENTA:
{tool_description}

CONTEXTO DEL SISTEMA:
- El sistema analiza datos financieros (crypto, forex, acciones)
- Los datos vienen en formato pandas DataFrame con columnas: timestamp, open, high, low, close, volume
- Las funciones deben ser eficientes y manejar casos edge
- El resultado debe ser un valor numérico o diccionario con métricas

REQUISITOS:
1. Genera una función Python completa y funcional
2. Incluye documentación clara con docstrings
3. Maneja errores y casos edge apropiadamente
4. Usa bibliotecas estándar (pandas, numpy, math) cuando sea posible
5. La función debe recibir un DataFrame como parámetro principal
6. Retorna un resultado que pueda ser usado por el sistema de análisis

FORMATO DE RESPUESTA:
```python
import pandas as pd
import numpy as np

def nueva_funcion_analisis(df: pd.DataFrame, **kwargs) -> float:
    \"\"\"
    Descripción de la función

    Args:
        df: DataFrame con datos OHLCV
        **kwargs: Parámetros adicionales

    Returns:
        Resultado del análisis
    \"\"\"
    # Código aquí

    return resultado
```

GENERA LA FUNCIÓN:"""

    async def _call_deepseek_api(self, prompt: str) -> Optional[str]:
        """Llamar a la API de Deepseek."""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Eres un ingeniero de software senior experto en Python y análisis financiero."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self._log(f"❌ Error llamando API Deepseek: {e}")
            return None

    def _process_repair_response(self, response: str, file_path: str, 
                               error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar respuesta de reparación de Deepseek."""
        try:
            # Extraer código de la respuesta
            code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)

            if code_blocks:
                fixed_code = code_blocks[0].strip()

                # Extraer explicación
                explanation = response.split('EXPLICACIÓN:')[-1].strip() if 'EXPLICACIÓN:' in response else 'Corrección automática aplicada'

                return {
                    'success': True,
                    'fixed_code': fixed_code,
                    'explanation': explanation,
                    'file_path': file_path,
                    'error_info': error_info,
                    'ready_for_approval': True,
                    'confidence': 0.85
                }
            else:
                return {
                    'success': False,
                    'error': 'No se pudo extraer código corregido de la respuesta'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error procesando respuesta: {str(e)}'
            }

    def _process_tool_response(self, response: str, tool_description: str) -> Dict[str, Any]:
        """Procesar respuesta de generación de herramientas."""
        try:
            # Extraer código de la respuesta
            code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)

            if code_blocks:
                generated_code = code_blocks[0].strip()

                # Extraer nombre de función
                function_match = re.search(r'def (\w+)\(', generated_code)
                function_name = function_match.group(1) if function_match else 'new_analyzer'

                return {
                    'success': True,
                    'code': generated_code,
                    'function_name': function_name,
                    'description': tool_description
                }
            else:
                return {
                    'success': False,
                    'error': 'No se pudo extraer código de la respuesta'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error procesando herramienta: {str(e)}'
            }

    def _simulate_repair(self, error_info: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Simular reparación cuando no hay API key."""
        return {
            'success': True,
            'fixed_code': f"# Corrección simulada para {error_info['error_type']}\n# TODO: Implementar corrección real",
            'explanation': f"Corrección simulada para {error_info['error_type']} en {file_path}",
            'file_path': file_path,
            'error_info': error_info,
            'ready_for_approval': True,
            'confidence': 0.5,
            'simulated': True
        }

    def _simulate_tool_generation(self, tool_description: str) -> Dict[str, Any]:
        """Simular generación de herramientas cuando no hay API key."""
        tool_id = f"simulated_tool_{len(self.generated_tools) + 1}"

        simulated_code = f'''def nueva_funcion_analisis(df, **kwargs):
    """
    Función simulada: {tool_description}
    """
    # TODO: Implementar lógica real
    return 0.5  # Valor placeholder
'''

        return {
            'success': True,
            'tool_id': tool_id,
            'tool_data': {
                'description': tool_description,
                'code': simulated_code,
                'created_at': datetime.now().isoformat(),
                'function_name': 'nueva_funcion_analisis'
            },
            'ready_for_approval': True,
            'simulated': True
        }

    def get_repair_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de reparaciones."""
        return self.repair_history

    def get_generated_tools(self) -> Dict[str, Any]:
        """Obtener herramientas generadas."""
        return self.generated_tools

    def apply_tool_to_system(self, tool_id: str) -> Dict[str, Any]:
        """Aplicar herramienta generada al sistema."""
        if tool_id not in self.generated_tools:
            return {
                'success': False,
                'error': f'Herramienta {tool_id} no encontrada'
            }

        try:
            tool_data = self.generated_tools[tool_id]

            # Guardar herramienta en archivo
            tools_dir = 'dynamic_tools'
            os.makedirs(tools_dir, exist_ok=True)

            tool_file = os.path.join(tools_dir, f"{tool_id}.py")
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(f"""# Herramienta generada dinámicamente por Deepseek Engineer
# Descripción: {tool_data['description']}
# Creada: {tool_data['created_at']}

{tool_data['code']}
""")

            self._log(f"✅ Herramienta {tool_id} guardada en {tool_file}")

            return {
                'success': True,
                'tool_id': tool_id,
                'file_path': tool_file,
                'function_name': tool_data['function_name']
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error aplicando herramienta: {str(e)}'
            }


# Instancia global del ingeniero
_deepseek_engineer = None

def get_deepseek_engineer(socketio=None):
    """Obtener instancia global del ingeniero Deepseek."""
    global _deepseek_engineer
    if _deepseek_engineer is None:
        _deepseek_engineer = DeepseekEngineer(socketio)
    return _deepseek_engineer

def handle_system_error(error_traceback: str, file_path: str):
    """Manejador global de errores del sistema."""
    engineer = get_deepseek_engineer()
    if engineer:
        # Ejecutar reparación en segundo plano
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(engineer.self_heal_on_error(error_traceback, file_path))
        except RuntimeError:
            # No hay loop activo, crear uno nuevo
            asyncio.run(engineer.self_heal_on_error(error_traceback, file_path))

# Instancia global del ingeniero para compatibilidad con el resto del sistema
deepseek_engineer = get_deepseek_engineer()
