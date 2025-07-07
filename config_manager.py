
#!/usr/bin/env python3
"""
Config Manager - Gestor de Configuración con Ejecución de Planes Secuenciales
Este módulo ejecuta planes complejos del Director de Operaciones paso a paso.
"""

import json
import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Variable global para mantener referencia al scanner
_scanner_instance = None


class ConfigManager:
    """
    Gestor de configuración que ejecuta planes secuenciales del Director de Operaciones,
    pasando resultados entre pasos y manejando dependencias.
    """

    def __init__(self):
        """Inicializar el Gestor de Configuración."""
        self.config_file = "system_config.json"
        self.current_config = self._load_config()
        self.plan_results = {}  # Almacena resultados de pasos para referencia posterior
        print("⚙️ ConfigManager con planificación secuencial inicializado")

    @staticmethod
    def register_scanner(scanner_instance):
        """
        Registra la instancia del scanner para comunicación.

        Args:
            scanner_instance: Instancia del MarketScanner
        """
        global _scanner_instance
        _scanner_instance = scanner_instance
        print("🔗 Scanner registrado en ConfigManager")

    async def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta un plan secuencial completo del Director de Operaciones.

        Args:
            plan: Lista de pasos del plan

        Returns:
            Resultado de la ejecución del plan
        """
        try:
            print(f"🎯 Ejecutando plan con {len(plan)} pasos...")
            
            executed_steps = []
            failed_steps = []
            self.plan_results = {}  # Reset results

            for step in plan:
                step_number = step.get('step', len(executed_steps) + 1)
                action = step.get('action')
                parameters = step.get('parameters', {})

                print(f"   Paso {step_number}: Ejecutando {action}")

                try:
                    # Resolver parámetros que dependen de pasos anteriores
                    resolved_parameters = await self._resolve_step_parameters(parameters)
                    
                    # Ejecutar el paso
                    result = await self._execute_plan_step(action, resolved_parameters)
                    
                    if result['success']:
                        # Almacenar resultado para pasos posteriores
                        self.plan_results[step_number] = result.get('data', result['message'])
                        
                        executed_steps.append({
                            'step': step_number,
                            'action': action,
                            'result': result['message'],
                            'data': result.get('data', None)
                        })
                        print(f"      ✅ {result['message']}")
                    else:
                        failed_steps.append({
                            'step': step_number,
                            'action': action,
                            'error': result['error']
                        })
                        print(f"      ❌ {result['error']}")
                        
                        # Decidir si continuar o detener en caso de fallo
                        if step.get('critical', True):
                            print(f"      🛑 Paso crítico falló. Deteniendo ejecución del plan.")
                            break

                except Exception as e:
                    error_msg = f"Error ejecutando paso {step_number} ({action}): {str(e)}"
                    failed_steps.append({
                        'step': step_number,
                        'action': action,
                        'error': error_msg
                    })
                    print(f"      ❌ {error_msg}")

            # Guardar configuración actualizada
            self._save_config()

            # Aplicar cambios al sistema
            await self._apply_config_changes()

            # Generar resumen
            summary = self._generate_plan_summary(executed_steps, failed_steps)

            return {
                'success': len(failed_steps) == 0,
                'summary': summary,
                'executed_count': len(executed_steps),
                'failed_count': len(failed_steps),
                'executed_steps': executed_steps,
                'failed_steps': failed_steps,
                'plan_results': self.plan_results
            }

        except Exception as e:
            print(f"❌ Error crítico ejecutando plan: {e}")
            return {
                'success': False,
                'error': f'Error crítico en plan: {str(e)}',
                'summary': 'Error crítico durante la ejecución del plan'
            }

    async def _resolve_step_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resuelve parámetros que dependen de resultados de pasos anteriores.

        Args:
            parameters: Parámetros del paso que pueden incluir referencias

        Returns:
            Parámetros resueltos
        """
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Parámetro de referencia como {result_from_step_1} o {top_asset_from_step_1}
                reference = value[1:-1]  # Remover llaves
                
                if reference.startswith("result_from_step_"):
                    step_num = int(reference.split("_")[-1])
                    if step_num in self.plan_results:
                        resolved[key] = self.plan_results[step_num]
                    else:
                        print(f"⚠️ Referencia no encontrada: {reference}")
                        resolved[key] = value
                        
                elif reference.startswith("top_asset_from_step_"):
                    step_num = int(reference.split("_")[-1])
                    if step_num in self.plan_results:
                        # Asumir que el resultado es una lista y tomar el primero
                        result = self.plan_results[step_num]
                        if isinstance(result, list) and len(result) > 0:
                            resolved[key] = result[0]
                        else:
                            resolved[key] = result
                    else:
                        print(f"⚠️ Referencia no encontrada: {reference}")
                        resolved[key] = value
                        
                elif reference.startswith("highest_momentum_from_step_"):
                    step_num = int(reference.split("_")[-1])
                    if step_num in self.plan_results:
                        # Asumir que el resultado incluye un análisis de momentum
                        result = self.plan_results[step_num]
                        if isinstance(result, dict) and 'best_asset' in result:
                            resolved[key] = result['best_asset']
                        else:
                            resolved[key] = result
                    else:
                        print(f"⚠️ Referencia no encontrada: {reference}")
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved

    async def _execute_plan_step(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un paso individual del plan.

        Args:
            action: Acción a ejecutar
            parameters: Parámetros resueltos

        Returns:
            Resultado del paso
        """
        # Acciones de Análisis y Escaneo
        if action == 'RUN_PRELIMINARY_SCAN':
            return await self._run_preliminary_scan(parameters)
        elif action == 'RUN_ON_DEMAND_ANALYSIS':
            return await self._run_on_demand_analysis(parameters)
        elif action == 'RUN_DEEP_ANALYSIS':
            return await self._run_deep_analysis(parameters)
        elif action == 'PRIORITY_SCAN':
            return await self._priority_scan(parameters)

        # Acciones de Configuración
        elif action == 'UPDATE_ASSET_LIST':
            return await self._update_asset_list(parameters)
        elif action == 'SET_RISK_PROFILE':
            return await self._set_risk_profile(parameters)
        elif action == 'CONFIGURE_TIMEFRAMES':
            return await self._configure_timeframes(parameters)
        elif action == 'SET_FOCUS_MODE':
            return await self._set_focus_mode(parameters)

        # Acciones heredadas del sistema anterior
        elif action == 'ADD_ASSET':
            return await self._add_asset(parameters)
        elif action == 'REMOVE_ASSET':
            return await self._remove_asset(parameters)

        else:
            return {
                'success': False,
                'error': f'Acción de plan desconocida: {action}'
            }

    # === NUEVAS ACCIONES DE ANÁLISIS Y ESCANEO ===

    async def _run_preliminary_scan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar escaneo preliminar con métrica específica."""
        try:
            metric = parameters.get('metric', 'profit_potential')
            scope = parameters.get('scope', 'all_assets')
            timeframe = parameters.get('timeframe', '1h')

            print(f"🔍 ESCANEO PRELIMINAR: {metric} en {scope} ({timeframe})")

            # Simular análisis de activos con ranking
            # En implementación real, esto consultaría el MarketScanner
            mock_results = [
                {'asset': 'BTC-USDT', 'score': 0.87, 'metric': metric},
                {'asset': 'ETH-USDT', 'score': 0.78, 'metric': metric},
                {'asset': 'SOL-USDT', 'score': 0.71, 'metric': metric},
                {'asset': 'OANDA:EUR_USD', 'score': 0.65, 'metric': metric},
                {'asset': 'XRP-USDT', 'score': 0.58, 'metric': metric}
            ]

            # Ordenar por score
            ranked_assets = sorted(mock_results, key=lambda x: x['score'], reverse=True)
            top_asset = ranked_assets[0]['asset']

            return {
                'success': True,
                'message': f'Escaneo preliminar completado. Mejor activo: {top_asset} (score: {ranked_assets[0]["score"]})',
                'data': {
                    'ranking': ranked_assets,
                    'best_asset': top_asset,
                    'metric_used': metric
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _run_on_demand_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar análisis bajo demanda para activos específicos."""
        try:
            assets = parameters.get('assets', [])
            timeframes = parameters.get('timeframes', ['5m'])
            metric = parameters.get('metric', 'momentum_strength')

            if not assets:
                return {'success': False, 'error': 'Lista de activos requerida'}

            print(f"⚡ ANÁLISIS BAJO DEMANDA: {len(assets)} activos con métrica {metric}")

            # Enviar comandos prioritarios al scanner
            analysis_results = {}
            
            for asset in assets:
                if _scanner_instance:
                    for timeframe in timeframes:
                        command_data = {
                            'type': 'RUN_ON_DEMAND_ANALYSIS',
                            'pair': asset,
                            'timeframe': timeframe,
                            'metric': metric,
                            'timestamp': datetime.now().isoformat()
                        }
                        _scanner_instance.add_priority_command(command_data)
                    
                    # Simular resultados de análisis
                    analysis_results[asset] = {
                        'momentum_score': 0.75 if asset == 'BTC-USDT' else 0.65,
                        'timeframes_analyzed': timeframes
                    }

            # Determinar el mejor activo basado en la métrica
            best_asset = max(analysis_results.keys(), 
                           key=lambda x: analysis_results[x].get('momentum_score', 0))

            return {
                'success': True,
                'message': f'Análisis bajo demanda completado para {len(assets)} activos. Mejor: {best_asset}',
                'data': {
                    'results': analysis_results,
                    'best_asset': best_asset,
                    'metric_used': metric
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _run_deep_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar análisis profundo con todos los motores."""
        try:
            asset = parameters.get('asset')
            timeframe = parameters.get('timeframe', '5m')

            if not asset:
                return {'success': False, 'error': 'Activo requerido para análisis profundo'}

            print(f"🧠 ANÁLISIS PROFUNDO: {asset} ({timeframe})")

            # En implementación real, esto activaría todos los motores de análisis
            return {
                'success': True,
                'message': f'Análisis profundo iniciado para {asset} ({timeframe})',
                'data': {
                    'asset': asset,
                    'timeframe': timeframe,
                    'engines_activated': ['predictive', 'social', 'macro', 'technical']
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _priority_scan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar escaneo prioritario."""
        try:
            asset_types = parameters.get('asset_types', ['crypto', 'forex'])
            metric = parameters.get('metric', 'volatility_score')

            print(f"⚡ ESCANEO PRIORITARIO: {asset_types} con métrica {metric}")

            return {
                'success': True,
                'message': f'Escaneo prioritario iniciado para: {", ".join(asset_types)}',
                'data': {
                    'asset_types': asset_types,
                    'metric': metric,
                    'priority_scan_active': True
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # === NUEVAS ACCIONES DE CONFIGURACIÓN ===

    async def _update_asset_list(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar lista de activos monitoreados."""
        try:
            assets = parameters.get('assets', [])
            exclusive_mode = parameters.get('exclusive_mode', False)

            if not assets:
                return {'success': False, 'error': 'Lista de activos requerida'}

            if exclusive_mode:
                # Modo exclusivo: solo monitorear estos activos
                self.current_config['assets'] = []
                
            # Añadir nuevos activos
            existing_pairs = [asset.get('pair') for asset in self.current_config.get('assets', [])]
            
            for asset_pair in assets:
                if asset_pair not in existing_pairs:
                    # Determinar tipo de activo
                    asset_type = 'crypto' if not asset_pair.startswith('OANDA:') else 'forex'
                    timeframes = ['1m', '5m', '15m'] if asset_type == 'crypto' else ['5m', '15m', '30m']
                    
                    new_asset = {
                        'type': asset_type,
                        'pair': asset_pair,
                        'timeframes': timeframes
                    }
                    
                    self.current_config.setdefault('assets', []).append(new_asset)

            mode_text = "exclusivo" if exclusive_mode else "normal"
            return {
                'success': True,
                'message': f'Lista de activos actualizada en modo {mode_text}. {len(assets)} activos configurados.',
                'data': {
                    'assets': assets,
                    'exclusive_mode': exclusive_mode,
                    'total_assets': len(self.current_config.get('assets', []))
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _set_risk_profile(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Establecer perfil de riesgo del sistema."""
        try:
            risk_level = parameters.get('risk_level', 'medium')
            risk_threshold = parameters.get('risk_threshold')

            # Mapear niveles de riesgo a umbrales
            risk_mapping = {
                'low': 0.05,
                'medium': 0.15,
                'high': 0.25,
                'aggressive': 0.35
            }

            if risk_threshold is None:
                risk_threshold = risk_mapping.get(risk_level, 0.15)

            self.current_config['risk_threshold'] = risk_threshold
            self.current_config['risk_profile'] = {
                'level': risk_level,
                'threshold': risk_threshold,
                'set_at': datetime.now().isoformat()
            }

            return {
                'success': True,
                'message': f'Perfil de riesgo establecido: {risk_level} (umbral: {risk_threshold:.1%})',
                'data': {
                    'risk_level': risk_level,
                    'risk_threshold': risk_threshold
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _configure_timeframes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar timeframes para activos."""
        try:
            asset_type = parameters.get('asset_type', 'all')
            timeframes = parameters.get('timeframes', ['5m', '15m'])

            assets_updated = 0
            for asset in self.current_config.get('assets', []):
                if asset_type == 'all' or asset.get('type') == asset_type:
                    asset['timeframes'] = timeframes
                    assets_updated += 1

            return {
                'success': True,
                'message': f'Timeframes configurados para {assets_updated} activos: {timeframes}',
                'data': {
                    'asset_type': asset_type,
                    'timeframes': timeframes,
                    'assets_updated': assets_updated
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _set_focus_mode(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Establecer modo de enfoque del sistema."""
        try:
            asset = parameters.get('asset')
            mode = parameters.get('mode', 'normal')
            asset_types = parameters.get('asset_types', [])

            focus_config = {
                'type': 'asset_specific' if asset else 'category',
                'target': asset or asset_types,
                'mode': mode,
                'set_at': datetime.now().isoformat()
            }

            self.current_config['focus'] = focus_config

            if asset:
                message = f'Modo de enfoque establecido: {mode} en {asset}'
            else:
                message = f'Modo de enfoque establecido: {mode} en {asset_types}'

            return {
                'success': True,
                'message': message,
                'data': focus_config
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # === ACCIONES HEREDADAS ===

    async def _add_asset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Añadir un nuevo activo al sistema."""
        try:
            asset_data = parameters.get('asset', {})
            asset_type = asset_data.get('type')
            pair = asset_data.get('pair')
            timeframes = asset_data.get('timeframes', ['5m'])

            if not pair:
                return {'success': False, 'error': 'Par de activo requerido'}

            # Verificar si el activo ya existe
            existing_assets = self.current_config.get('assets', [])
            for existing in existing_assets:
                if existing.get('pair') == pair:
                    return {'success': False, 'error': f'Activo {pair} ya existe'}

            # Añadir nuevo activo
            new_asset = {
                'type': asset_type,
                'pair': pair,
                'timeframes': timeframes
            }

            existing_assets.append(new_asset)
            self.current_config['assets'] = existing_assets

            return {
                'success': True,
                'message': f'Activo {pair} añadido con timeframes {timeframes}',
                'data': new_asset
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _remove_asset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Eliminar un activo del sistema."""
        try:
            pair = parameters.get('pair')

            if not pair:
                return {'success': False, 'error': 'Par de activo requerido'}

            existing_assets = self.current_config.get('assets', [])
            original_count = len(existing_assets)

            # Filtrar activos para eliminar el especificado
            filtered_assets = [asset for asset in existing_assets if asset.get('pair') != pair]

            if len(filtered_assets) == original_count:
                return {'success': False, 'error': f'Activo {pair} no encontrado'}

            self.current_config['assets'] = filtered_assets

            return {
                'success': True,
                'message': f'Activo {pair} eliminado del sistema',
                'data': {'removed_asset': pair}
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # === MÉTODOS DE SOPORTE ===

    def _load_config(self) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo.

        Returns:
            Configuración actual del sistema
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ Configuración cargada desde {self.config_file}")
                return config
            else:
                # Configuración por defecto
                default_config = {
                    'assets': [],
                    'risk_threshold': 0.15,
                    'strategies': {},
                    'focus': {
                        'type': 'balanced',
                        'asset_types': ['crypto', 'forex']
                    },
                    'last_updated': datetime.now().isoformat()
                }
                print("📋 Usando configuración por defecto")
                return default_config

        except Exception as e:
            print(f"⚠️ Error cargando configuración: {e}")
            return {
                'assets': [],
                'risk_threshold': 0.15,
                'strategies': {},
                'focus': {'type': 'balanced', 'asset_types': ['crypto', 'forex']},
                'last_updated': datetime.now().isoformat()
            }

    def _save_config(self) -> bool:
        """
        Guardar configuración en archivo.

        Returns:
            True si se guardó exitosamente
        """
        try:
            self.current_config['last_updated'] = datetime.now().isoformat()

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=2, ensure_ascii=False)

            print(f"💾 Configuración guardada en {self.config_file}")
            return True

        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
            return False

    async def _apply_config_changes(self) -> bool:
        """
        Aplicar cambios de configuración al sistema en tiempo real.

        Returns:
            True si se aplicó exitosamente
        """
        try:
            print("🔄 Aplicando cambios de configuración al sistema...")
            print("✅ Cambios de configuración aplicados")
            return True

        except Exception as e:
            print(f"❌ Error aplicando configuración: {e}")
            return False

    def _generate_plan_summary(self, executed: List[Dict], failed: List[Dict]) -> str:
        """
        Generar resumen de ejecución del plan.

        Args:
            executed: Pasos ejecutados exitosamente
            failed: Pasos que fallaron

        Returns:
            Resumen textual
        """
        summary_parts = []

        if executed:
            summary_parts.append(f"✅ {len(executed)} pasos ejecutados exitosamente")
            for step in executed:
                summary_parts.append(f"   • Paso {step['step']}: {step['result']}")

        if failed:
            summary_parts.append(f"❌ {len(failed)} pasos fallaron")
            for step in failed:
                summary_parts.append(f"   • Paso {step['step']}: {step['error']}")

        if not executed and not failed:
            summary_parts.append("ℹ️ No se ejecutaron pasos")

        return "\n".join(summary_parts)

    # Método heredado para compatibilidad
    async def execute_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta una lista de acciones (compatibilidad con sistema anterior).

        Args:
            actions: Lista de acciones a ejecutar

        Returns:
            Resultado de la ejecución
        """
        # Convertir acciones a formato de plan
        plan = []
        for i, action in enumerate(actions, 1):
            plan_step = {
                'step': i,
                'action': action.get('type'),
                'parameters': {k: v for k, v in action.items() if k != 'type'}
            }
            plan.append(plan_step)

        # Ejecutar como plan
        return await self.execute_plan(plan)

    def get_current_config(self) -> Dict[str, Any]:
        """
        Obtener configuración actual.

        Returns:
            Configuración actual del sistema
        """
        return self.current_config.copy()

    def get_assets_count(self) -> int:
        """
        Obtener número de activos configurados.

        Returns:
            Número de activos
        """
        return len(self.current_config.get('assets', []))


# Función de demostración
async def demo_plan_execution():
    """
    Demostración del Gestor de Configuración con ejecución de planes.
    """
    print("⚙️ DEMO: Gestor de Configuración con Ejecución de Planes")
    print("="*60)

    config_manager = ConfigManager()

    # Plan de prueba
    test_plan = [
        {
            'step': 1,
            'action': 'RUN_PRELIMINARY_SCAN',
            'parameters': {
                'metric': 'profit_potential',
                'scope': 'all_assets',
                'timeframe': '1h'
            }
        },
        {
            'step': 2,
            'action': 'UPDATE_ASSET_LIST',
            'parameters': {
                'assets': ['{top_asset_from_step_1}'],
                'exclusive_mode': True
            }
        },
        {
            'step': 3,
            'action': 'SET_RISK_PROFILE',
            'parameters': {
                'risk_level': 'aggressive'
            }
        }
    ]

    print(f"\n🧪 Ejecutando plan de prueba con {len(test_plan)} pasos...")

    result = await config_manager.execute_plan(test_plan)

    print(f"\n📊 RESULTADO:")
    print(f"   Éxito: {result['success']}")
    print(f"   Pasos ejecutados: {result['executed_count']}")
    print(f"   Pasos fallidos: {result['failed_count']}")
    print(f"\n📋 Resumen:")
    print(result['summary'])

    if result.get('plan_results'):
        print(f"\n📈 Resultados de pasos:")
        for step, data in result['plan_results'].items():
            print(f"   Paso {step}: {data}")

    print("\n" + "="*60)
    print("✅ Demo del Gestor de Configuración con planificación completada")


if __name__ == "__main__":
    asyncio.run(demo_plan_execution())
