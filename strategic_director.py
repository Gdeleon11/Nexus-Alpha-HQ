#!/usr/bin/env python3
"""
Strategic Director - Director de Operaciones con Capacidades de Planificación
Este módulo traduce comandos complejos del CEO en planes secuenciales de acciones.
"""

import os
import json
import asyncio
from openai import OpenAI
from typing import Dict, List, Any
from deepseek_engineer import get_deepseek_engineer


class StrategicDirector:
    """
    Director de Operaciones que interpreta comandos complejos del CEO
    y los descompone en planes secuenciales de acciones técnicas.
    """

    def __init__(self):
        """Inicializar el Director de Operaciones con GPT-4."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("🎯 Director de Operaciones con planificación inicializado con GPT-4")
        else:
            self.openai_client = None
            print("⚠️ OPENAI_API_KEY no encontrada. Director funcionará en modo limitado.")

    async def interpret_command(self, command_text: str) -> Dict[str, Any]:
        """
        Interpreta un comando complejo del CEO y genera un plan secuencial.

        Args:
            command_text: Comando del CEO

        Returns:
            Diccionario con el plan de ejecución
        """
        if not self.openai_client:
            return {
                'success': False,
                'error': 'Director de Operaciones no disponible - OpenAI API no configurada',
                'natural_response': 'Lo siento, el sistema de planificación no está disponible en este momento.'
            }

        try:
            print(f"🧠 Planificando ejecución del comando: {command_text}")

            # Meta-Prompt de planificación
            meta_prompt = self._create_planning_meta_prompt()

            # Llamar a GPT-4 para planificar
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": meta_prompt},
                    {"role": "user", "content": command_text}
                ],
                max_tokens=2000,
                temperature=0.1
            )

            # Procesar la respuesta
            interpretation = response.choices[0].message.content.strip()

            try:
                # Extraer JSON de la respuesta
                json_start = interpretation.find('{')
                json_end = interpretation.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = interpretation[json_start:json_end]
                    parsed_data = json.loads(json_str)
                else:
                    raise ValueError("No se encontró JSON válido en la respuesta")

                print(f"✅ Plan generado exitosamente:")
                print(f"   Pasos del plan: {len(parsed_data.get('plan', []))}")
                print(f"   Razonamiento: {parsed_data.get('reasoning', 'No disponible')}")

                return {
                    'success': True,
                    'plan': parsed_data.get('plan', []),
                    'reasoning': parsed_data.get('reasoning', 'Plan generado'),
                    'natural_response': parsed_data.get('response_message', 'Plan ejecutándose...'),
                    'response_message': parsed_data.get('response_message', 'Entendido. Ejecutando plan paso a paso...'),
                    'raw_response': interpretation
                }

            except json.JSONDecodeError as e:
                print(f"❌ Error parseando JSON de GPT-4: {e}")
                print(f"Respuesta cruda: {interpretation}")

                return {
                    'success': False,
                    'error': f'Error interpretando comando: {str(e)}',
                    'natural_response': 'Disculpa, no pude entender tu solicitud. ¿Podrías reformularla?',
                    'raw_response': interpretation
                }

        except Exception as e:
            print(f"❌ Error en Director de Operaciones: {e}")
            return {
                'success': False,
                'error': f'Error procesando comando: {str(e)}',
                'natural_response': 'Ha ocurrido un error interno. Inténtalo de nuevo en unos momentos.'
            }

    def _create_planning_meta_prompt(self) -> str:
        """
        Crea el meta-prompt de planificación para GPT-4.

        Returns:
            Meta-prompt de planificación
        """
        return """Eres el Director de Operaciones de Nexus-Alpha, un sistema de trading automatizado avanzado. Recibes órdenes del CEO y tu trabajo es pensar paso a paso y descomponer cada orden en un plan secuencial de acciones técnicas.

**TU MISIÓN:**
Analizar el comando del CEO y crear un plan detallado que el sistema pueda ejecutar paso a paso. Cada paso puede usar el resultado del paso anterior.

**ACCIONES DISPONIBLES:**

**ANÁLISIS Y ESCANEO:**
- `RUN_PRELIMINARY_SCAN`: Escanear todos los activos con una métrica específica
- `RUN_ON_DEMAND_ANALYSIS`: Análizar un activo específico inmediatamente
- `RUN_DEEP_ANALYSIS`: Análisis profundo de un activo usando todos los motores
- `PRIORITY_SCAN`: Escaneo prioritario de tipos de activos específicos

**CONFIGURACIÓN:**
- `UPDATE_ASSET_LIST`: Actualizar lista de activos monitoreados
- `SET_RISK_PROFILE`: Cambiar perfil de riesgo del sistema
- `CONFIGURE_TIMEFRAMES`: Configurar timeframes para activos
- `SET_FOCUS_MODE`: Establecer modo de enfoque (crypto, forex, específico)

**MÉTRICAS DISPONIBLES:**
- `profit_potential`: Potencial de rentabilidad
- `volatility_score`: Score de volatilidad
- `momentum_strength`: Fuerza del momentum
- `risk_reward_ratio`: Relación riesgo-beneficio
- `technical_score`: Score técnico combinado

**PROCESO DE RAZONAMIENTO:**
1. **Analiza** el comando del CEO e identifica las partes clave
2. **Descompón** en pasos lógicos secuenciales
3. **Identifica** qué datos necesita cada paso
4. **Planifica** cómo el resultado de un paso alimenta al siguiente
5. **Genera** el plan JSON final

**FORMATO DE RESPUESTA:**
Responde ÚNICAMENTE con un JSON válido en este formato:

```json
{
  "reasoning": "Explicación paso a paso de tu razonamiento interno",
  "plan": [
    {
      "step": 1,
      "action": "TIPO_ACCION",
      "parameters": {
        "param1": "valor1",
        "param2": "valor2"
      },
      "expected_output": "Descripción de lo que este paso debería producir",
      "feeds_to_step": 2
    },
    {
      "step": 2,
      "action": "TIPO_ACCION",
      "parameters": {
        "input_from_step": 1,
        "param1": "{result_from_step_1}"
      },
      "expected_output": "Resultado final esperado"
    }
  ],
  "response_message": "Respuesta conversacional para el CEO"
}
```

**EJEMPLOS:**

**Comando CEO:** "Encuentra el activo más rentable hoy y enfócate solo en él"

**Tu Razonamiento Interno:**
1. El CEO quiere identificar UN activo con mayor potencial
2. Primero necesito escanear TODOS los activos para medir rentabilidad
3. Luego debo reconfigurar el sistema para enfocarse solo en ese activo
4. El resultado del paso 1 (mejor activo) alimenta al paso 2 (reconfiguración)

**Tu Respuesta:**
```json
{
  "reasoning": "El CEO solicita identificar y enfocar en el activo más rentable. Esto requiere: 1) Escaneo completo para ranking de rentabilidad, 2) Reconfiguración del sistema para monitoreo exclusivo del ganador.",
  "plan": [
    {
      "step": 1,
      "action": "RUN_PRELIMINARY_SCAN",
      "parameters": {
        "metric": "profit_potential",
        "scope": "all_assets",
        "timeframe": "1h"
      },
      "expected_output": "Ranking de activos por potencial de rentabilidad",
      "feeds_to_step": 2
    },
    {
      "step": 2,
      "action": "UPDATE_ASSET_LIST",
      "parameters": {
        "input_from_step": 1,
        "assets": ["{top_asset_from_step_1}"],
        "exclusive_mode": true
      },
      "expected_output": "Sistema reconfigurado para monitoreo exclusivo"
    }
  ],
  "response_message": "Entendido, CEO. Iniciando escaneo completo para identificar el activo más prometedor. Luego reconfiguré el sistema para enfoque total en ese activo."
}
```

**Comando CEO:** "Analiza Bitcoin y Ethereum en timeframes cortos, luego configura el sistema para trading agresivo en el que tenga mejor momentum"

**Tu Respuesta:**
```json
{
  "reasoning": "El CEO quiere comparar BTC y ETH en timeframes cortos, identificar cuál tiene mejor momentum, y configurar trading agresivo en el ganador. Plan: 1) Análisis paralelo de ambos, 2) Configuración de modo agresivo en el ganador.",
  "plan": [
    {
      "step": 1,
      "action": "RUN_ON_DEMAND_ANALYSIS",
      "parameters": {
        "assets": ["BTC-USDT", "ETH-USDT"],
        "timeframes": ["1m", "5m"],
        "metric": "momentum_strength"
      },
      "expected_output": "Scores de momentum para BTC y ETH",
      "feeds_to_step": 2
    },
    {
      "step": 2,
      "action": "SET_FOCUS_MODE",
      "parameters": {
        "input_from_step": 1,
        "asset": "{highest_momentum_from_step_1}",
        "mode": "aggressive_trading"
      },
      "expected_output": "Sistema configurado para trading agresivo"
    }
  ],
  "response_message": "Perfecto. Analizando momentum de BTC y ETH en timeframes cortos. Configuré modo agresivo en el que muestre mayor fuerza."
}
```

RESPONDE ÚNICAMENTE CON JSON VÁLIDO. Piensa paso a paso como un director de operaciones que debe traducir órdenes ejecutivas en acciones técnicas precisas."""

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de acciones disponibles para planificación.

        Returns:
            Lista de acciones disponibles
        """
        return [
            {
                'category': 'ANÁLISIS',
                'actions': [
                    'RUN_PRELIMINARY_SCAN',
                    'RUN_ON_DEMAND_ANALYSIS',
                    'RUN_DEEP_ANALYSIS',
                    'PRIORITY_SCAN'
                ]
            },
            {
                'category': 'CONFIGURACIÓN',
                'actions': [
                    'UPDATE_ASSET_LIST',
                    'SET_RISK_PROFILE',
                    'CONFIGURE_TIMEFRAMES',
                    'SET_FOCUS_MODE'
                ]
            }
        ]

    def get_available_metrics(self) -> List[str]:
        """
        Obtiene la lista de métricas disponibles.

        Returns:
            Lista de métricas disponibles
        """
        return [
            'profit_potential',
            'volatility_score',
            'momentum_strength',
            'risk_reward_ratio',
            'technical_score'
        ]


# Función de demostración
async def demo_planning_director():
    """
    Demostración del Director de Operaciones con planificación.
    """
    print("🎯 DEMO: Director de Operaciones con Planificación")
    print("="*70)

    director = StrategicDirector()

    # Comandos de prueba complejos
    test_commands = [
        "Encuentra el activo más rentable hoy y enfócate solo en él",
        "Analiza Bitcoin y Ethereum en timeframes cortos, luego configura el sistema para trading agresivo en el que tenga mejor momentum",
        "Escanea todos los pares de forex, encuentra los 3 con mayor volatilidad y configura el sistema para operar solo esos",
        "Reduce el riesgo al 10% y enfócate en criptomonedas de alta capitalización"
    ]

    for i, command in enumerate(test_commands, 1):
        print(f"\n🧪 PRUEBA {i}: {command}")
        print("-" * 70)

        result = await director.interpret_command(command)

        if result['success']:
            print(f"🧠 Razonamiento: {result['reasoning']}")
            print(f"📋 Plan con {len(result['plan'])} pasos:")
            for j, step in enumerate(result['plan'], 1):
                print(f"   Paso {j}: {step['action']}")
                print(f"     Parámetros: {step.get('parameters', {})}")
                print(f"     Resultado esperado: {step.get('expected_output', 'N/A')}")
            print(f"💬 Respuesta: {result['natural_response']}")
        else:
            print(f"❌ Error: {result['error']}")
            print(f"💬 Respuesta: {result['natural_response']}")

    print("\n" + "="*70)
    print("✅ Demo del Director de Operaciones con planificación completada")


# Standalone wrapper functions for import compatibility
def interpret_command(command_text: str):
    """
    Standalone wrapper function for interpreting commands.
    
    Args:
        command_text: Command text to interpret
        
    Returns:
        Dictionary with interpretation result
    """
    director = StrategicDirector()
    
    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(director.interpret_command(command_text))
        return result
    finally:
        loop.close()

def execute_command(plan, priority_queue, config_manager):
    """
    Execute a strategic plan by processing its steps.
    
    Args:
        plan: Plan dictionary from interpret_command
        priority_queue: Priority queue for urgent tasks
        config_manager: Configuration manager
        
    Returns:
        String response message
    """
    if not plan.get('success', False):
        return plan.get('natural_response', 'Error procesando comando.')
    
    # For now, return the natural response from the plan
    # TODO: Implement actual plan execution logic
    return plan.get('natural_response', 'Plan interpretado correctamente.')

if __name__ == "__main__":
    asyncio.run(demo_planning_director())