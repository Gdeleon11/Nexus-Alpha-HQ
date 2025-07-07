#!/usr/bin/env python3
"""
Evolutionary Engine for DataNexus - Genetic Algorithm Strategy Optimization
Este módulo usa algoritmos genéticos con DEAP para evolucionar estrategias de trading óptimas.
"""

import random
import asyncio
import numpy as np
import sqlite3
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import pandas as pd

# Try to import DEAP, if not available, implement basic genetic algorithm
try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("Warning: DEAP not available, using basic genetic algorithm implementation")

from backtester import run_backtest
from datanexus_connector import DataNexus

class EvolutionaryEngine:
    """
    Motor evolutivo que usa algoritmos genéticos para encontrar estrategias de trading óptimas.
    """
    
    def __init__(self, data_connector: DataNexus):
        """
        Inicializa el motor evolutivo.
        
        Args:
            data_connector: Instancia de DataNexus para obtener datos históricos
        """
        self.data_connector = data_connector
        self.historical_data = None
        self.generation_count = 0
        self.best_strategies_history = []
        
        if HAS_DEAP:
            self._setup_deap_environment()
        else:
            self._setup_basic_ga()
    
    def _setup_deap_environment(self):
        """Configura el entorno DEAP para algoritmos genéticos."""
        # Limpiar cualquier definición previa
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Crear tipos de fitness e individuo
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        # Configurar toolbox
        self.toolbox = base.Toolbox()
        
        # Registrar funciones de generación
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("evaluate", self._evaluate_strategy)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _setup_basic_ga(self):
        """Configura algoritmo genético básico cuando DEAP no está disponible."""
        self.population = []
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def _create_individual(self) -> Dict[str, Any]:
        """
        Crea un individuo (estrategia) con genes aleatorios.
        Ahora incluye indicadores descubiertos por Alpha Genesis.
        
        Returns:
            Diccionario con los parámetros de la estrategia
        """
        # Indicadores tradicionales + Alpha Genesis
        traditional_indicators = ['RSI', 'MACD', 'SMA']
        alpha_genesis_indicators = self._get_alpha_genesis_indicators()
        
        all_indicators = traditional_indicators + alpha_genesis_indicators
        timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        strategy = {
            'indicator': random.choice(all_indicators),
            'timeframe': random.choice(timeframes),
        }
        
        # Parámetros específicos según el indicador
        if strategy['indicator'] == 'RSI':
            strategy.update({
                'entry_level': random.randint(15, 35),
                'exit_level': random.randint(60, 85),
                'rsi_period': random.randint(10, 20)
            })
        elif strategy['indicator'] == 'MACD':
            strategy.update({
                'fast_period': random.randint(8, 15),
                'slow_period': random.randint(20, 30),
                'signal_period': random.randint(7, 12)
            })
        elif strategy['indicator'] == 'SMA':
            fast = random.randint(5, 15)
            slow = random.randint(fast + 5, 30)
            strategy.update({
                'sma_fast': fast,
                'sma_slow': slow
            })
        elif strategy['indicator'].startswith('alpha_genesis'):
            # Indicador descubierto por Alpha Genesis
            alpha_config = self._get_alpha_genesis_config(strategy['indicator'])
            strategy.update(alpha_config)
        
        if HAS_DEAP:
            individual = creator.Individual(strategy)
        else:
            individual = strategy
            individual['fitness'] = 0.0
        
        return individual
    
    def _get_alpha_genesis_indicators(self):
        """
        Obtiene la lista de indicadores descubiertos por Alpha Genesis.
        
        Returns:
            Lista de nombres de indicadores Alpha Genesis
        """
        try:
            from alpha_genesis_engine import AlphaGenesisEngine
            alpha_genesis = AlphaGenesisEngine()
            
            # Obtener mejores indicadores
            best_indicators = alpha_genesis.get_best_indicators(5)  # Top 5
            
            # Filtrar indicadores con fitness mínimo
            alpha_indicators = []
            for indicator in best_indicators:
                if indicator['fitness'] > 0.5:  # Solo indicadores con buen fitness
                    alpha_indicators.append(indicator['id'])
            
            print(f"🧬 Alpha Genesis: {len(alpha_indicators)} indicadores disponibles para evolución")
            return alpha_indicators
            
        except Exception as e:
            print(f"⚠️ Error obteniendo indicadores Alpha Genesis: {str(e)}")
            return []
    
    def _get_alpha_genesis_config(self, indicator_id):
        """
        Obtiene la configuración de un indicador Alpha Genesis.
        
        Args:
            indicator_id: ID del indicador
            
        Returns:
            Diccionario con configuración del indicador
        """
        try:
            from alpha_genesis_engine import AlphaGenesisEngine
            alpha_genesis = AlphaGenesisEngine()
            
            config = alpha_genesis.export_indicator_for_evolution(indicator_id)
            if config:
                return {
                    'alpha_expression': config['expression'],
                    'alpha_fitness': config['fitness_score'],
                    'alpha_id': indicator_id,
                    'entry_threshold': random.uniform(-2.0, -0.5),
                    'exit_threshold': random.uniform(0.5, 2.0)
                }
            else:
                # Configuración por defecto si no se puede obtener
                return {
                    'alpha_expression': '(high + low) / 2 - close',
                    'alpha_fitness': 0.5,
                    'alpha_id': indicator_id,
                    'entry_threshold': -1.0,
                    'exit_threshold': 1.0
                }
                
        except Exception as e:
            print(f"Error obteniendo config Alpha Genesis: {str(e)}")
            return {
                'alpha_expression': '(high + low) / 2 - close',
                'alpha_fitness': 0.5,
                'alpha_id': indicator_id,
                'entry_threshold': -1.0,
                'exit_threshold': 1.0
            }
    
    def _get_collective_intelligence_performance(self, individual: Dict[str, Any]) -> Dict[str, float]:
        """
        Obtiene métricas de rendimiento desde la Red de Inteligencia Colectiva.
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Diccionario con métricas de la red colectiva
        """
        try:
            import requests
            
            # Extraer parámetros de la estrategia y mapear a formato colectivo
            indicator = individual.get('indicator', '').upper()
            timeframe = individual.get('timeframe', '')
            
            # Construir nombre de estrategia para búsqueda colectiva
            collective_strategy_patterns = []
            if indicator:
                # Buscar patrones de estrategia BUY y SELL
                collective_strategy_patterns.extend([
                    f"{indicator}_BUY",
                    f"{indicator}_SELL"
                ])
            
            best_collective_metric = None
            
            # Probar diferentes patrones de estrategia
            for strategy_pattern in collective_strategy_patterns:
                params = {
                    'estrategia': strategy_pattern,
                    'timeframe': timeframe
                }
                
                # Consultar métricas de Red de Inteligencia Colectiva
                response = requests.get(
                    "http://0.0.0.0:5000/api/collective_metrics",  # Mismo servidor, puerto 5000
                    params=params,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    collective_metrics = data.get('collective_metrics', [])
                    
                    if collective_metrics:
                        # Evaluar si esta métrica es mejor que la anterior
                        metric = collective_metrics[0]  # Ya están ordenadas por win_rate DESC
                        
                        if (best_collective_metric is None or 
                            metric['total_trades'] > best_collective_metric['total_trades'] or
                            (metric['total_trades'] >= best_collective_metric['total_trades'] and 
                             metric['win_rate'] > best_collective_metric['win_rate'])):
                            
                            best_collective_metric = metric
                            print(f"🌐 Red de Inteligencia - {strategy_pattern}-{timeframe}: {metric['wins']}W/{metric['losses']}L ({metric['win_rate']:.1f}%) - {metric['total_trades']} trades globales - {metric['contributing_nodes']} nodos")
            
            if best_collective_metric:
                return {
                    'win_rate': best_collective_metric['win_rate'] / 100.0,  # Convertir a decimal
                    'total_trades': best_collective_metric['total_trades'],
                    'wins': best_collective_metric['wins'],
                    'losses': best_collective_metric['losses'],
                    'confidence_score': best_collective_metric['confidence_score'],
                    'contributing_nodes': best_collective_metric['contributing_nodes'],
                    'is_collective': True
                }
            else:
                print(f"🌐 Red de Inteligencia - {indicator}-{timeframe}: Sin datos colectivos disponibles")
                return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0, 'confidence_score': 0.0, 'contributing_nodes': 0, 'is_collective': False}
                
        except requests.exceptions.RequestException:
            print(f"🔌 Red de Inteligencia no disponible - usando datos locales")
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0, 'confidence_score': 0.0, 'contributing_nodes': 0, 'is_collective': False}
        except Exception as e:
            print(f"❌ Error en Red de Inteligencia: {str(e)}")
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0, 'confidence_score': 0.0, 'contributing_nodes': 0, 'is_collective': False}

    def _get_real_trading_performance(self, individual: Dict[str, Any]) -> Dict[str, float]:
        """
        Obtiene métricas precisas de trading real desde trades.db para una estrategia específica.
        
        NUEVA IMPLEMENTACIÓN: Búsqueda más precisa y cálculo exacto del rendimiento real.
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Diccionario con métricas exactas de trading real
        """
        try:
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()
            
            # Extraer parámetros clave de la estrategia
            indicator = individual.get('indicator', '').upper()
            timeframe = individual.get('timeframe', '')
            
            print(f"🔍 Buscando historial real para estrategia: {indicator}-{timeframe}")
            
            # Búsqueda en trades.db por indicador y timeframe
            search_conditions = []
            search_params = []
            
            if indicator:
                # Buscar el indicador en el campo verdict
                search_conditions.append("UPPER(verdict) LIKE ?")
                search_params.append(f'%{indicator}%')
            
            if timeframe:
                # Buscar el timeframe en asset o verdict
                search_conditions.append("(UPPER(asset) LIKE ? OR UPPER(verdict) LIKE ?)")
                search_params.extend([f'%{timeframe}%', f'%{timeframe}%'])
            
            if not search_conditions:
                print(f"⚠️ No se pueden crear criterios de búsqueda para la estrategia")
                conn.close()
                return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}
            
            # Construir consulta SQL
            where_clause = ' AND '.join(search_conditions)
            query = f'''
                SELECT result, timestamp, asset, verdict 
                FROM trades 
                WHERE {where_clause} 
                AND result IN ('WIN', 'LOSS')
                ORDER BY timestamp DESC
            '''
            
            print(f"📋 Ejecutando consulta: {query}")
            print(f"📋 Parámetros: {search_params}")
            
            cursor.execute(query, search_params)
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                print(f"📊 Sin historial de trading real para {indicator}-{timeframe}")
                return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}
            
            # Calcular métricas exactas
            wins = sum(1 for trade in trades if trade[0] == 'WIN')
            losses = sum(1 for trade in trades if trade[0] == 'LOSS')
            total_trades = len(trades)
            
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            # Mostrar detalles de los trades encontrados
            print(f"📊 Historial encontrado para {indicator}-{timeframe}:")
            for i, trade in enumerate(trades[:3], 1):  # Mostrar primeros 3
                result_emoji = "🟢" if trade[0] == 'WIN' else "🔴"
                print(f"    {i}. {result_emoji} {trade[0]} - {trade[1]} - {trade[2]}")
            
            if len(trades) > 3:
                print(f"    ... y {len(trades) - 3} trades más")
            
            metrics = {
                'win_rate': win_rate,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses
            }
            
            print(f"📈 RESUMEN: {wins}W/{losses}L = {win_rate:.1%} en {total_trades} trades")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error obteniendo historial de trading real: {str(e)}")
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}
    
    def _get_real_trading_metrics(self, individual: Dict[str, Any]) -> Dict[str, float]:
        """
        Obtiene métricas completas de trading real desde el Diario de Trading.
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Diccionario con métricas de trading real
        """
        try:
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()
            
            # Obtener parámetros de la estrategia para búsqueda más precisa
            indicator = individual.get('indicator', '').upper()
            timeframe = individual.get('timeframe', '')
            
            # Búsqueda más específica por parámetros de estrategia
            search_patterns = []
            search_params = []
            
            if indicator and timeframe:
                # Buscar por indicador y timeframe en el veredicto
                search_patterns.append("UPPER(verdict) LIKE ? AND UPPER(verdict) LIKE ?")
                search_params.extend([f'%{indicator}%', f'%{timeframe}%'])
                
                # También buscar por asset que contenga el timeframe
                search_patterns.append("UPPER(asset) LIKE ? AND UPPER(verdict) LIKE ?")
                search_params.extend([f'%{timeframe}%', f'%{indicator}%'])
            
            if not search_patterns:
                conn.close()
                return {'win_rate': 0.0, 'total_trades': 0, 'confidence': 0.0}
            
            # Combinar patrones de búsqueda con OR
            query = f'''
                SELECT result, timestamp FROM trades 
                WHERE ({' OR '.join(search_patterns)}) AND
                      result IN ('WIN', 'LOSS')
                ORDER BY timestamp DESC
            '''
            
            cursor.execute(query, search_params)
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                print(f"📊 Estrategia {indicator}-{timeframe}: Sin historial de trading real")
                return {'win_rate': 0.0, 'total_trades': 0, 'confidence': 0.0}
            
            # Calcular métricas detalladas
            wins = sum(1 for trade in trades if trade[0] == 'WIN')
            losses = len(trades) - wins
            total_trades = len(trades)
            
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            # Calcular confianza basada en el número de trades
            # Más trades = mayor confianza en la métrica
            confidence = min(total_trades / 20.0, 1.0)  # Max confianza con 20+ trades
            
            # Penalización por racha de pérdidas recientes
            recent_penalty = 1.0
            if len(trades) >= 3:
                recent_trades = [trade[0] for trade in trades[:3]]  # Últimos 3 trades
                recent_losses = sum(1 for result in recent_trades if result == 'LOSS')
                if recent_losses >= 2:  # 2 o más pérdidas en últimos 3 trades
                    recent_penalty = 0.8
                    print(f"⚠️  Penalización por racha de pérdidas recientes: {recent_penalty}")
            
            metrics = {
                'win_rate': win_rate * recent_penalty,
                'total_trades': total_trades,
                'confidence': confidence,
                'wins': wins,
                'losses': losses,
                'recent_penalty': recent_penalty
            }
            
            print(f"📊 Estrategia {indicator}-{timeframe}: {wins}W/{losses}L ({win_rate:.1%}) - {total_trades} trades - Confianza: {confidence:.1%}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error obteniendo métricas de trading real: {str(e)}")
            return {'win_rate': 0.0, 'total_trades': 0, 'confidence': 0.0}
    
    def _evaluate_strategy(self, individual: Dict[str, Any]) -> Tuple[float,]:
        """
        Evalúa una estrategia basándose PRIORITARIAMENTE en los resultados reales del Diario de Trading.
        
        NUEVA METODOLOGÍA CON PRUEBAS DE ESTRÉS GAN:
        - 70% peso a resultados reales del Diario de Trading (trades.db)
        - 20% peso a pruebas de estrés con escenarios sintéticos GAN
        - 10% peso a backtesting teórico
        - PENALIZACIÓN SEVERA para estrategias con <40% win rate real
        - BONUS por robustez en escenarios sintéticos adversos
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Tuple con el fitness basado en trading real y robustez sintética
        """
        try:
            # 1. OBTENER HISTORIAL LOCAL DE TRADING REAL DESDE trades.db
            local_metrics = self._get_real_trading_performance(individual)
            local_win_rate = local_metrics['win_rate']
            total_local_trades = local_metrics['total_trades']
            local_wins = local_metrics['wins']
            local_losses = local_metrics['losses']
            
            # 2. OBTENER MÉTRICAS DE RED DE INTELIGENCIA COLECTIVA
            collective_metrics = self._get_collective_intelligence_performance(individual)
            collective_win_rate = collective_metrics['win_rate']
            total_collective_trades = collective_metrics['total_trades']
            collective_confidence = collective_metrics['confidence_score']
            is_collective_available = collective_metrics['is_collective']
            
            print(f"📊 Estrategia {individual.get('indicator', 'Unknown')}-{individual.get('timeframe', 'Unknown')}")
            print(f"    📈 Local: {local_wins}W/{local_losses}L = {local_win_rate:.1%} ({total_local_trades} trades)")
            if is_collective_available:
                print(f"    🌐 Red Colectiva: {collective_metrics['wins']}W/{collective_metrics['losses']}L = {collective_win_rate:.1%} ({total_collective_trades} trades)")
            
            # 3. COMBINAR MÉTRICAS LOCAL Y COLECTIVA CON PONDERACIÓN INTELIGENTE
            if is_collective_available and total_collective_trades >= 10:
                # Ajustar peso colectivo basado en número de nodos contribuyentes
                node_factor = min(collective_metrics.get('contributing_nodes', 1) / 5.0, 1.0)
                collective_weight = min(collective_confidence * 0.8 * node_factor, 0.8)  # Máximo 80% peso colectivo
                local_weight = 1.0 - collective_weight
                
                # Calcular win rate combinado con ponderación inteligente
                combined_win_rate = (local_win_rate * local_weight) + (collective_win_rate * collective_weight)
                combined_total_trades = total_local_trades + total_collective_trades
                
                # Bonus por diversidad de la red (más nodos = más confiable)
                network_diversity_bonus = min(collective_metrics.get('contributing_nodes', 1) / 10.0, 0.1)
                combined_win_rate += network_diversity_bonus
                
                print(f"    🔄 Combinado: {combined_win_rate:.1%} (Local {local_weight:.1%} + Colectivo {collective_weight:.1%})")
                print(f"    🌐 Red diversa: {collective_metrics.get('contributing_nodes', 0)} nodos, bonus: +{network_diversity_bonus:.1%}")
                
                # Usar métricas combinadas
                real_win_rate = combined_win_rate
                total_real_trades = combined_total_trades
                wins = local_wins + collective_metrics['wins']
                losses = local_losses + collective_metrics['losses']
                
            elif total_local_trades > 0:
                # Solo datos locales disponibles
                real_win_rate = local_win_rate
                total_real_trades = total_local_trades
                wins = local_wins
                losses = local_losses
                print(f"    📊 Usando solo datos locales")
                
            else:
                # Sin datos locales, intentar usar solo colectivos si están disponibles
                if is_collective_available:
                    real_win_rate = collective_win_rate * 0.8  # Descuento por no tener validación local
                    total_real_trades = total_collective_trades
                    wins = collective_metrics['wins']
                    losses = collective_metrics['losses']
                    print(f"    🌐 Usando solo datos colectivos (con descuento)")
                else:
                    real_win_rate = 0.0
                    total_real_trades = 0
                    wins = 0
                    losses = 0
                    print(f"    ❌ Sin datos disponibles (local ni colectivo)")
            
            # 2. PRUEBA DE ESTRÉS CON ESCENARIOS SINTÉTICOS GAN
            stress_test_fitness = self._get_stress_test_fitness(individual)
            print(f"    🧪 Fitness de prueba de estrés: {stress_test_fitness:.3f}")
            
            # 3. CALCULAR FITNESS CON NUEVA PONDERACIÓN: 70% REAL / 20% STRESS TEST / 10% BACKTEST
            if total_real_trades >= 3:  # Mínimo 3 trades para considerar historial válido
                
                # COMPONENTE REAL (70% del fitness)
                real_fitness_base = real_win_rate  # Win rate como base (0.0 - 1.0)
                
                # APLICAR PENALIZACIONES Y BONIFICACIONES SEVERAS
                if real_win_rate < 0.4:  # Menos del 40% - PENALIZACIÓN SEVERA
                    penalty_factor = 0.1  # Reducir fitness al 10%
                    real_fitness_base *= penalty_factor
                    print(f"    🚨 PENALIZACIÓN SEVERA: Win rate {real_win_rate:.1%} < 40% - Fitness reducido al 10%")
                    
                elif real_win_rate >= 0.7:  # 70% o más - BONIFICACIÓN ALTA
                    bonus_factor = 1.8  # Aumentar fitness 80%
                    real_fitness_base *= bonus_factor
                    print(f"    🚀 BONIFICACIÓN ALTA: Win rate {real_win_rate:.1%} >= 70% - Fitness aumentado 80%")
                    
                elif real_win_rate >= 0.6:  # 60-69% - BONIFICACIÓN MODERADA
                    bonus_factor = 1.4  # Aumentar fitness 40%
                    real_fitness_base *= bonus_factor
                    print(f"    ✅ BONIFICACIÓN MODERADA: Win rate {real_win_rate:.1%} >= 60% - Fitness aumentado 40%")
                
                else:  # 40-59% - Sin modificación especial
                    print(f"    ⚖️ RANGO NEUTRAL: Win rate {real_win_rate:.1%} - Sin bonificación/penalización")
                
                # Bonificación por volumen de trades exitosos
                if total_real_trades >= 10:
                    volume_bonus = min(total_real_trades / 20.0, 0.3)  # Hasta 30% bonus por volumen
                    real_fitness_base += volume_bonus
                    print(f"    📈 BONUS VOLUMEN: {total_real_trades} trades - Bonus: +{volume_bonus:.1%}")
                
                # COMPONENTE BACKTEST (10% del fitness)
                backtest_fitness = self._get_backtest_fitness(individual)
                
                # CÁLCULO FINAL: 70% real + 20% stress test + 10% backtest
                final_fitness = (real_fitness_base * 0.7) + (stress_test_fitness * 0.2) + (backtest_fitness * 0.1)
                
                print(f"    🎯 FITNESS FINAL: Real({real_fitness_base:.3f})*70% + Stress({stress_test_fitness:.3f})*20% + Backtest({backtest_fitness:.3f})*10% = {final_fitness:.3f}")
                
            elif total_real_trades > 0:  # Historial muy limitado (1-2 trades)
                # Dar peso moderado a los pocos resultados reales
                limited_real_fitness = real_win_rate * 0.5  # Descuento por poca muestra
                backtest_fitness = self._get_backtest_fitness(individual)
                
                # 50% real limitado + 30% stress test + 20% backtest
                final_fitness = (limited_real_fitness * 0.5) + (stress_test_fitness * 0.3) + (backtest_fitness * 0.2)
                
                print(f"    ⚠️ HISTORIAL MUY LIMITADO: {total_real_trades} trades")
                print(f"    🎯 FITNESS: Real limitado({limited_real_fitness:.3f})*50% + Stress({stress_test_fitness:.3f})*30% + Backtest({backtest_fitness:.3f})*20% = {final_fitness:.3f}")
                
            else:  # Sin historial real - PENALIZACIÓN MÁXIMA
                # Stress test + backtest con penalización severa
                backtest_fitness = self._get_backtest_fitness(individual) * 0.1  # Solo 10% del backtest normal
                final_fitness = (stress_test_fitness * 0.7) + (backtest_fitness * 0.3)
                
                print(f"    🔍 SIN HISTORIAL REAL - PENALIZACIÓN MÁXIMA")
                print(f"    🎯 FITNESS: Stress({stress_test_fitness:.3f})*70% + Backtest penalizado({backtest_fitness:.3f})*30% = {final_fitness:.3f}")
            
            # Asegurar rango válido [0.0, 2.0]
            final_fitness = max(min(final_fitness, 2.0), 0.0)
            
            return (final_fitness,)
            
        except Exception as e:
            print(f"❌ Error evaluando estrategia: {str(e)}")
            return (0.0,)
    
    def _get_stress_test_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Calcula fitness basado en pruebas de estrés con escenarios sintéticos GAN.
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Fitness de prueba de estrés normalizado (0.0 - 1.0)
        """
        try:
            # Importar el simulador GAN
            from gan_simulator import GANMarketSimulator, stress_test_strategy
            
            # Verificar si existe modelo GAN entrenado
            import os
            model_path = 'models/timegan_model.pkl'
            
            if not os.path.exists(model_path):
                print(f"    🤖 Modelo GAN no disponible, entrenando rápidamente...")
                # Entrenar modelo GAN básico si no existe
                if self.historical_data is not None and len(self.historical_data) >= 100:
                    simulator = GANMarketSimulator()
                    simulator.train_market_gan(
                        historical_data=self.historical_data,
                        sequence_length=50,  # Reducido para velocidad
                        epochs=10  # Entrenamiento mínimo
                    )
                else:
                    print(f"    ⚠️ Datos insuficientes para entrenar GAN")
                    return self._get_backtest_fitness(individual) * 0.3
            
            # Ejecutar prueba de estrés con escenarios sintéticos
            stress_results = stress_test_strategy(individual, num_scenarios=30)  # Optimizado para velocidad
            
            if 'error' in stress_results or stress_results['successful_tests'] == 0:
                print(f"    🧪 Stress test falló - fitness: 0.0")
                return 0.0
            
            # Extraer métricas clave
            success_rate = stress_results['success_rate']
            robustness_score = stress_results.get('robustness_score', 0) / 100  # Normalizar a [0,1]
            
            if 'win_rate_stats' in stress_results:
                avg_win_rate = stress_results['win_rate_stats']['mean'] / 100  # Normalizar a [0,1]
                win_rate_std = stress_results['win_rate_stats']['std'] / 100  # Normalizar
                
                # Penalizar alta variabilidad (preferir consistencia)
                consistency_bonus = max(0, 1 - win_rate_std * 2)
                
                # Bonus por performance en condiciones adversas
                if avg_win_rate > 0.6 and robustness_score > 0.7:
                    adversity_bonus = 0.2  # 20% bonus por robustez excepcional
                else:
                    adversity_bonus = 0.0
            else:
                avg_win_rate = 0.5
                consistency_bonus = 0.5
                adversity_bonus = 0.0
            
            # Calcular fitness compuesto de prueba de estrés
            stress_fitness = (
                success_rate * 0.25 +          # 25% tasa de éxito en escenarios
                robustness_score * 0.35 +      # 35% score de robustez
                avg_win_rate * 0.25 +          # 25% win rate promedio
                consistency_bonus * 0.1 +      # 10% bonus por consistencia
                adversity_bonus * 0.05         # 5% bonus por robustez excepcional
            )
            
            stress_fitness = max(min(stress_fitness, 1.0), 0.0)
            
            print(f"    🧪 GAN Stress Test: {stress_results['successful_tests']}/{stress_results['total_scenarios']} éxito, robustez: {robustness_score:.2f}, fitness: {stress_fitness:.3f}")
            
            return stress_fitness
            
        except Exception as e:
            print(f"    ⚠️ Error en GAN stress test (usando fallback): {str(e)}")
            # Fallback a fitness básico si GAN no está disponible
            return self._get_backtest_fitness(individual) * 0.4  # Penalizar moderadamente
    
    def _get_backtest_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Calcula fitness del backtest (componente secundario).
        
        Args:
            individual: Estrategia a evaluar
            
        Returns:
            Fitness del backtest normalizado
        """
        try:
            if self.historical_data is None or len(self.historical_data) < 50:
                return 0.0
            
            results = run_backtest(individual, self.historical_data)
            
            if 'error' in results or results['total_trades'] == 0:
                return 0.0
            
            # Métricas normalizadas del backtest
            profit_factor = min(results.get('profit_factor', 0), 3.0) / 3.0  # Normalizar a [0,1]
            win_rate = results.get('win_rate', 0) / 100  # Convertir a decimal
            max_drawdown = min(results.get('max_drawdown', 100), 100) / 100
            
            # Penalizar si hay muy pocas operaciones
            trade_penalty = min(results['total_trades'] / 20, 1.0)
            
            backtest_fitness = profit_factor * win_rate * (1 - max_drawdown) * trade_penalty
            
            return min(backtest_fitness, 1.0)
            
        except Exception as e:
            print(f"Error en backtest fitness: {str(e)}")
            return 0.0
    
    def _crossover(self, ind1: Dict[str, Any], ind2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Cruza dos individuos para crear descendencia.
        
        Args:
            ind1, ind2: Individuos padre
            
        Returns:
            Tuple con dos individuos hijo
        """
        if ind1.get('indicator') == ind2.get('indicator'):
            # Mismo indicador, cruzar parámetros comunes
            common_keys = set(ind1.keys()) & set(ind2.keys())
            for key in common_keys:
                if key not in ['indicator', 'timeframe', 'fitness'] and isinstance(ind1.get(key), (int, float)):
                    if random.random() < 0.5:
                        ind1[key], ind2[key] = ind2[key], ind1[key]
        else:
            # Diferentes indicadores, crear nuevos individuos aleatorios
            if random.random() < 0.5:
                ind1_new = self._create_individual()
                ind2_new = self._create_individual()
                # Conservar algunos parámetros originales
                ind1_new['timeframe'] = ind1.get('timeframe', '1h')
                ind2_new['timeframe'] = ind2.get('timeframe', '1h')
                # Actualizar individuos
                for key in ind1_new.keys():
                    if key != 'fitness':
                        ind1[key] = ind1_new[key]
                for key in ind2_new.keys():
                    if key != 'fitness':
                        ind2[key] = ind2_new[key]
        
        return ind1, ind2
    
    def _mutate(self, individual: Dict[str, Any]) -> Tuple[Dict[str, Any],]:
        """
        Muta un individuo.
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Tuple con el individuo mutado
        """
        if random.random() < 0.1:  # 10% chance to change indicator
            indicators = ['RSI', 'MACD', 'SMA']
            new_indicator = random.choice([i for i in indicators if i != individual['indicator']])
            individual = self._create_individual()
            individual['indicator'] = new_indicator
        
        # Mutar parámetros numéricos
        for key, value in individual.items():
            if isinstance(value, int) and key != 'indicator':
                if random.random() < 0.1:  # 10% chance to mutate each parameter
                    if 'period' in key:
                        individual[key] = max(5, min(30, value + random.randint(-3, 3)))
                    elif 'level' in key:
                        if 'entry' in key:
                            individual[key] = max(10, min(40, value + random.randint(-5, 5)))
                        else:  # exit level
                            individual[key] = max(50, min(90, value + random.randint(-5, 5)))
                    else:
                        individual[key] = max(1, value + random.randint(-2, 2))
        
        return (individual,)
    
    def _ensure_database_exists(self):
        """
        Asegura que la base de datos de trades existe y tiene la estructura correcta.
        """
        try:
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()
            
            # Crear tabla si no existe
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    signal_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    asset TEXT,
                    verdict TEXT,
                    result TEXT DEFAULT 'PENDING'
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error ensuring database exists: {str(e)}")
    
    async def prepare_data(self, pair: str = 'BTCUSDT', timeframe: str = '5m', limit: int = 200) -> bool:
        """
        Prepara datos históricos para el entrenamiento.
        
        Args:
            pair: Par de trading
            limit: Número de velas históricas
            
        Returns:
            True si los datos se obtuvieron correctamente
        """
        try:
            print(f"📊 Obteniendo datos históricos para {pair}...")
            
            if pair.startswith('OANDA:'):
                # Mapear timeframe a resolución de forex
                resolution_map = {
                    '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                    '1h': '60', '4h': '240', '1d': 'D'
                }
                forex_resolution = resolution_map.get(timeframe, '5')
                self.historical_data = await self.data_connector.get_forex_data(pair, forex_resolution, limit)
            else:
                self.historical_data = await self.data_connector.get_crypto_data(pair, timeframe, limit)
            
            if self.historical_data is None or len(self.historical_data) < 50:
                print(f"❌ Error: Datos insuficientes para {pair}")
                return False
            
            print(f"✅ Datos obtenidos: {len(self.historical_data)} velas")
            return True
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {str(e)}")
            return False
    
    async def evolve_best_strategy(self, generations: int = 20, population_size: int = 50, pair: str = 'BTCUSDT', timeframe: str = '5m') -> Optional[Dict[str, Any]]:
        """
        Evoluciona la mejor estrategia priorizando FUERTEMENTE los resultados de trading real.
        
        NUEVA METODOLOGÍA:
        - 90% peso a resultados reales del Diario de Trading
        - 10% peso a backtesting histórico
        - Estrategias ganadoras reales dominan la evolución
        - Estrategias perdedoras son eliminadas agresivamente
        
        Args:
            generations: Número de generaciones
            population_size: Tamaño de la población
            pair: Par de trading para entrenar
            timeframe: Timeframe para entrenar
            
        Returns:
            Mejor estrategia basada en performance real
        """
        print("🧬 EVOLUCIÓN CON RED DE INTELIGENCIA COLECTIVA - APRENDIZAJE DISTRIBUIDO")
        print("="*80)
        print("🎯 METODOLOGÍA: Supervivencia basada en experiencia local + sabiduría colectiva")
        print("📊 PONDERACIÓN: 70% experiencia local + 30% Red de Inteligencia Colectiva + backtesting")
        print("🌐 RED COLECTIVA: Acceso a experiencias de trading de toda la red Nexus-Alpha")
        print("🚨 PENALIZACIONES: Estrategias con <40% win rate eliminadas agresivamente")
        print("🚀 OBJETIVO: Estrategias validadas por la inteligencia colectiva global")
        print("="*80)
        
        # Asegurar que la base de datos existe
        self._ensure_database_exists()
        
        # Verificar si hay datos de trading real disponibles (local + colectivo)
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE result IN ('WIN', 'LOSS')")
        local_trades_count = cursor.fetchone()[0]
        conn.close()
        
        # Verificar datos de Red de Inteligencia Colectiva
        collective_trades_count = 0
        try:
            import requests
            response = requests.get("http://0.0.0.0:5001/api/collective_metrics", timeout=3)
            if response.status_code == 200:
                data = response.json()
                collective_trades_count = sum(m['total_trades'] for m in data.get('collective_metrics', []))
        except:
            pass
        
        print(f"✅ DATOS LOCALES: {local_trades_count} trades históricos")
        print(f"🌐 RED COLECTIVA: {collective_trades_count} trades de la red global")
        
        total_intelligence = local_trades_count + collective_trades_count
        if total_intelligence > 0:
            print(f"🧠 INTELIGENCIA TOTAL: {total_intelligence} experiencias de trading disponibles")
            print("🔥 Evolución potenciada por sabiduría colectiva")
        else:
            print("⚠️ SIN DATOS DE INTELIGENCIA - Evolución basada en backtest únicamente")
        
        # Preparar datos históricos para backtest (componente secundario)
        if not await self.prepare_data(pair, timeframe):
            print("📉 Datos históricos limitados - Foco total en experiencia de trading real")
        
        if HAS_DEAP:
            return await self._evolve_with_deap(generations, population_size)
        else:
            return await self._evolve_basic(generations, population_size)
    
    async def _evolve_with_deap(self, generations: int, population_size: int) -> Optional[Dict[str, Any]]:
        """Evolución usando DEAP con enfoque en trading real."""
        print(f"🔬 DEAP - Algoritmo Genético con Supervivencia por Trading Real")
        print(f"👥 Población: {population_size} estrategias candidatas")
        print(f"🔄 Generaciones: {generations} ciclos evolutivos")
        print(f"📊 NUEVA PONDERACIÓN: 90% trading real + 10% backtest")
        print(f"⚡ Selección natural: Solo estrategias ganadoras reales sobreviven")
        
        # Crear población inicial
        population = self.toolbox.population(n=population_size)
        
        # Evaluar población inicial
        print("🧪 Evaluando población inicial...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        best_fitness_history = []
        
        for generation in range(generations):
            print(f"\n🔄 Generación {generation + 1}/{generations}")
            
            # Selección
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover y mutación
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:  # Probabilidad de crossover
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:  # Probabilidad de mutación
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluar individuos invalidados
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Reemplazar población
            population[:] = offspring
            
            # Estadísticas
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness = max(fits)
            avg_fitness = sum(fits) / len(fits)
            best_fitness_history.append(best_fitness)
            
            print(f"   📊 Mejor fitness: {best_fitness:.4f}")
            print(f"   📊 Fitness promedio: {avg_fitness:.4f}")
            
            # Mostrar mejor estrategia actual
            best_ind = max(population, key=lambda x: x.fitness.values[0])
            print(f"   🏆 Mejor estrategia: {dict(best_ind)}")
        
        # Obtener mejor individuo final
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        final_fitness = best_individual.fitness.values[0]
        
        print("\n🎉 EVOLUCIÓN COMPLETADA")
        print("="*60)
        print(f"🏆 Mejor fitness final: {final_fitness:.4f}")
        print(f"🧬 Mejor estrategia: {dict(best_individual)}")
        
        return dict(best_individual)
    
    async def _evolve_basic(self, generations: int, population_size: int) -> Optional[Dict[str, Any]]:
        """Algoritmo genético básico con enfoque en trading real."""
        print(f"🔬 Algoritmo Genético Básico - Aprendizaje de Trading Real")
        print(f"👥 Población: {population_size} estrategias en competencia")
        print(f"🔄 Generaciones: {generations} rondas de selección natural")
        print(f"📊 NUEVA PONDERACIÓN: 90% experiencia real + 10% simulación")
        print(f"🎯 Supervivencia: Solo las estrategias rentables reales prosperan")
        
        # Crear población inicial
        population = [self._create_individual() for _ in range(population_size)]
        
        # Evaluar población inicial
        print("🧪 Evaluando población inicial...")
        for individual in population:
            fitness = self._evaluate_strategy(individual)
            individual['fitness'] = fitness[0]
        
        for generation in range(generations):
            print(f"\n🔄 Generación {generation + 1}/{generations}")
            
            # Selección por torneo
            new_population = []
            for _ in range(population_size):
                tournament = random.sample(population, 3)
                winner = max(tournament, key=lambda x: x['fitness'])
                new_population.append(winner.copy())
            
            # Crossover y mutación
            for i in range(0, len(new_population) - 1, 2):
                if random.random() < self.crossover_rate:
                    self._crossover(new_population[i], new_population[i + 1])
                
                if random.random() < self.mutation_rate:
                    self._mutate(new_population[i])
                if random.random() < self.mutation_rate:
                    self._mutate(new_population[i + 1])
            
            # Evaluar nueva población
            for individual in new_population:
                fitness = self._evaluate_strategy(individual)
                individual['fitness'] = fitness[0]
            
            population = new_population
            
            # Estadísticas
            best_fitness = max(ind['fitness'] for ind in population)
            avg_fitness = sum(ind['fitness'] for ind in population) / len(population)
            
            print(f"   📊 Mejor fitness: {best_fitness:.4f}")
            print(f"   📊 Fitness promedio: {avg_fitness:.4f}")
            
            # Mostrar mejor estrategia actual
            best_ind = max(population, key=lambda x: x['fitness'])
            strategy_copy = best_ind.copy()
            strategy_copy.pop('fitness', None)
            print(f"   🏆 Mejor estrategia: {strategy_copy}")
        
        # Obtener mejor individuo final
        best_individual = max(population, key=lambda x: x['fitness'])
        final_fitness = best_individual['fitness']
        
        print("\n🎉 EVOLUCIÓN COMPLETADA")
        print("="*60)
        print(f"🏆 Mejor fitness final: {final_fitness:.4f}")
        
        # Limpiar fitness del resultado
        result = best_individual.copy()
        result.pop('fitness', None)
        print(f"🧬 Mejor estrategia: {result}")
        
        return result

# Función de conveniencia
async def evolve_trading_strategy(pair: str = 'BTCUSDT', timeframe: str = '5m', generations: int = 20, population_size: int = 50) -> Optional[Dict[str, Any]]:
    """
    Función principal para evolucionar una estrategia de trading.
    
    Args:
        pair: Par de trading
        generations: Número de generaciones
        population_size: Tamaño de la población
        
    Returns:
        Mejor estrategia encontrada
    """
    data_nexus = DataNexus()
    evolution_engine = EvolutionaryEngine(data_nexus)
    
    return await evolution_engine.evolve_best_strategy(generations, population_size, pair, timeframe)

# Función de demostración
async def demo_evolution():
    """
    Demostración del motor evolutivo.
    """
    print("🧬 DEMO: Evolución de Estrategias de Trading")
    print("="*60)
    
    try:
        # Evolucionar estrategia para BTC
        best_strategy = await evolve_trading_strategy('BTCUSDT', '5m', generations=10, population_size=30)
        
        if best_strategy:
            print("\n🎯 ESTRATEGIA EVOLUCIONADA EXITOSAMENTE")
            print("="*60)
            print(f"📊 Estrategia óptima: {best_strategy}")
            
            # Probar la estrategia evolucionada
            print("\n🧪 Probando estrategia evolucionada...")
            data_nexus = DataNexus()
            test_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 100)
            
            if test_data is not None:
                results = run_backtest(best_strategy, test_data)
                print(f"📈 Resultados del test:")
                print(f"   • Operaciones totales: {results['total_trades']}")
                print(f"   • Tasa de éxito: {results['win_rate']:.2f}%")
                print(f"   • Factor de ganancia: {results['profit_factor']:.2f}")
                print(f"   • Retorno total: {results['total_return']:.2f}%")
        else:
            print("❌ No se pudo evolucionar una estrategia válida")
            
    except Exception as e:
        print(f"❌ Error durante la evolución: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_evolution())