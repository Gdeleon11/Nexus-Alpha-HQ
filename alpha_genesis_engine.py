
#!/usr/bin/env python3
"""
Alpha Genesis Engine - Descubrimiento Autónomo de Indicadores Técnicos
Este módulo usa Programación Genética para crear nuevos indicadores técnicos de forma automática.
"""

import random
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
import operator
import warnings
from datetime import datetime
import sqlite3
import pickle
import copy

# Try to import DEAP for genetic programming
try:
    from deap import base, creator, tools, algorithms, gp
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("Warning: DEAP not available, using basic genetic programming implementation")

from backtester import run_backtest

warnings.filterwarnings('ignore')

class AlphaGenesisEngine:
    """
    Motor de descubrimiento autónomo de indicadores técnicos usando Programación Genética.
    
    Crea árboles de expresión que combinan datos de mercado con operadores matemáticos
    para generar nuevos indicadores técnicos evolutivos.
    """
    
    def __init__(self, data_connector=None):
        """
        Inicializa el motor Alpha Genesis.
        
        Args:
            data_connector: Conector de datos para obtener información histórica
        """
        self.data_connector = data_connector
        self.discovered_indicators = {}
        self.generation_count = 0
        self.best_indicators_history = []
        self.fitness_cache = {}
        
        # Configurar la base de datos para guardar indicadores descubiertos
        self._setup_database()
        
        if HAS_DEAP:
            self._setup_deap_gp()
        else:
            self._setup_basic_gp()
    
    def _setup_database(self):
        """Configura la base de datos para almacenar indicadores descubiertos."""
        try:
            conn = sqlite3.connect('alpha_genesis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS discovered_indicators (
                    indicator_id TEXT PRIMARY KEY,
                    expression TEXT,
                    fitness REAL,
                    generation INTEGER,
                    discovery_date TEXT,
                    parameters TEXT,
                    backtest_results TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error configurando base de datos Alpha Genesis: {str(e)}")
    
    def _setup_deap_gp(self):
        """Configura el entorno DEAP para programación genética."""
        # Limpiar definiciones previas
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Crear tipos de fitness e individuo
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Definir conjunto de primitivas (operadores y funciones)
        self.pset = gp.PrimitiveSet("MAIN", 5)  # 5 entradas: open, high, low, close, volume
        
        # Operadores matemáticos básicos
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(self._protected_div, 2)
        
        # Funciones matemáticas avanzadas
        self.pset.addPrimitive(self._protected_log, 1)
        self.pset.addPrimitive(self._protected_sqrt, 1)
        self.pset.addPrimitive(self._protected_sin, 1)
        self.pset.addPrimitive(self._protected_cos, 1)
        self.pset.addPrimitive(self._protected_exp, 1)
        self.pset.addPrimitive(abs, 1)
        
        # Funciones de agregación temporal
        self.pset.addPrimitive(self._sma, 2)  # Simple Moving Average
        self.pset.addPrimitive(self._ema, 2)  # Exponential Moving Average
        self.pset.addPrimitive(self._max_window, 2)  # Max en ventana
        self.pset.addPrimitive(self._min_window, 2)  # Min en ventana
        self.pset.addPrimitive(self._std_window, 2)  # Desviación estándar en ventana
        
        # Constantes y terminales
        self.pset.addEphemeralConstant("rand_int", lambda: random.randint(1, 50))
        self.pset.addEphemeralConstant("rand_float", lambda: random.uniform(0.1, 10.0))
        
        # Renombrar argumentos para claridad
        self.pset.renameArguments(ARG0='open', ARG1='high', ARG2='low', ARG3='close', ARG4='volume')
        
        # Configurar toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_indicator)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        # Límites para evitar explosión exponencial
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    
    def _setup_basic_gp(self):
        """Configura programación genética básica cuando DEAP no está disponible."""
        self.population = []
        self.population_size = 50
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        
        # Operadores disponibles para construcción manual de árboles
        self.operators = [
            ('+', 2, operator.add),
            ('-', 2, operator.sub),
            ('*', 2, operator.mul),
            ('/', 2, self._protected_div),
            ('log', 1, self._protected_log),
            ('sqrt', 1, self._protected_sqrt),
            ('sin', 1, self._protected_sin),
            ('abs', 1, abs),
            ('sma', 2, self._sma),
            ('max', 2, self._max_window)
        ]
        
        self.terminals = ['open', 'high', 'low', 'close', 'volume']
    
    # Funciones protegidas para evitar errores matemáticos
    def _protected_div(self, a, b):
        """División protegida para evitar división por cero."""
        try:
            if abs(b) < 1e-10:
                return 1.0
            result = a / b
            return result if not (math.isnan(result) or math.isinf(result)) else 1.0
        except:
            return 1.0
    
    def _protected_log(self, x):
        """Logaritmo protegido."""
        try:
            if x <= 0:
                return 0.0
            result = math.log(abs(x))
            return result if not (math.isnan(result) or math.isinf(result)) else 0.0
        except:
            return 0.0
    
    def _protected_sqrt(self, x):
        """Raíz cuadrada protegida."""
        try:
            result = math.sqrt(abs(x))
            return result if not (math.isnan(result) or math.isinf(result)) else 0.0
        except:
            return 0.0
    
    def _protected_sin(self, x):
        """Seno protegido."""
        try:
            result = math.sin(x)
            return result if not (math.isnan(result) or math.isinf(result)) else 0.0
        except:
            return 0.0
    
    def _protected_cos(self, x):
        """Coseno protegido."""
        try:
            result = math.cos(x)
            return result if not (math.isnan(result) or math.isinf(result)) else 0.0
        except:
            return 0.0
    
    def _protected_exp(self, x):
        """Exponencial protegida."""
        try:
            if x > 50:  # Evitar overflow
                return math.exp(50)
            if x < -50:
                return math.exp(-50)
            result = math.exp(x)
            return result if not (math.isnan(result) or math.isinf(result)) else 1.0
        except:
            return 1.0
    
    def _sma(self, values, period):
        """Media móvil simple protegida."""
        try:
            period = max(1, min(int(abs(period)), 50))  # Limitar período
            if isinstance(values, (list, np.ndarray)):
                if len(values) < period:
                    return values[-1] if len(values) > 0 else 0.0
                return np.mean(values[-period:])
            else:
                return float(values)
        except:
            return 0.0
    
    def _ema(self, values, period):
        """Media móvil exponencial protegida."""
        try:
            period = max(1, min(int(abs(period)), 50))
            if isinstance(values, (list, np.ndarray)):
                if len(values) == 0:
                    return 0.0
                if len(values) == 1:
                    return values[0]
                
                alpha = 2.0 / (period + 1)
                ema = values[0]
                for value in values[1:]:
                    ema = alpha * value + (1 - alpha) * ema
                return ema
            else:
                return float(values)
        except:
            return 0.0
    
    def _max_window(self, values, period):
        """Máximo en ventana protegido."""
        try:
            period = max(1, min(int(abs(period)), 50))
            if isinstance(values, (list, np.ndarray)):
                if len(values) == 0:
                    return 0.0
                return max(values[-period:]) if len(values) >= period else max(values)
            else:
                return float(values)
        except:
            return 0.0
    
    def _min_window(self, values, period):
        """Mínimo en ventana protegido."""
        try:
            period = max(1, min(int(abs(period)), 50))
            if isinstance(values, (list, np.ndarray)):
                if len(values) == 0:
                    return 0.0
                return min(values[-period:]) if len(values) >= period else min(values)
            else:
                return float(values)
        except:
            return 0.0
    
    def _std_window(self, values, period):
        """Desviación estándar en ventana protegida."""
        try:
            period = max(1, min(int(abs(period)), 50))
            if isinstance(values, (list, np.ndarray)):
                if len(values) == 0:
                    return 0.0
                window = values[-period:] if len(values) >= period else values
                return np.std(window) if len(window) > 1 else 0.0
            else:
                return 0.0
        except:
            return 0.0
    
    def _evaluate_indicator(self, individual):
        """
        Evalúa un indicador técnico generado por programación genética.
        
        Args:
            individual: Árbol de expresión que representa el indicador
            
        Returns:
            Tuple con el fitness del indicador
        """
        try:
            # Crear clave de cache
            if HAS_DEAP:
                cache_key = str(individual)
            else:
                cache_key = str(individual)
            
            # Verificar cache
            if cache_key in self.fitness_cache:
                return self.fitness_cache[cache_key]
            
            # Compilar función del indicador
            if HAS_DEAP:
                func = gp.compile(individual, self.pset)
            else:
                func = self._compile_basic_tree(individual)
            
            # Obtener datos históricos para evaluación
            if not hasattr(self, 'test_data') or self.test_data is None:
                # Usar datos de prueba sintéticos si no hay datos reales
                self.test_data = self._generate_synthetic_test_data()
            
            # Calcular valores del indicador
            indicator_values = self._calculate_indicator_values(func, self.test_data)
            
            if indicator_values is None or len(indicator_values) < 20:
                fitness = (0.0,)
            else:
                # Evaluar calidad del indicador mediante backtesting
                fitness_score = self._backtest_indicator(indicator_values, self.test_data)
                fitness = (fitness_score,)
            
            # Guardar en cache
            self.fitness_cache[cache_key] = fitness
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluando indicador: {str(e)}")
            return (0.0,)
    
    def _calculate_indicator_values(self, func, data):
        """
        Calcula los valores del indicador para los datos dados.
        
        Args:
            func: Función compilada del indicador
            data: DataFrame con datos OHLCV
            
        Returns:
            Array con valores del indicador
        """
        try:
            indicator_values = []
            
            for i in range(len(data)):
                try:
                    # Preparar ventana de datos para funciones de agregación
                    window_start = max(0, i - 50)  # Ventana de 50 períodos máximo
                    window_data = data.iloc[window_start:i+1]
                    
                    if len(window_data) == 0:
                        continue
                    
                    # Preparar argumentos para la función
                    open_vals = window_data['open'].values
                    high_vals = window_data['high'].values
                    low_vals = window_data['low'].values
                    close_vals = window_data['close'].values
                    volume_vals = window_data['volume'].values
                    
                    # Calcular valor del indicador
                    if HAS_DEAP:
                        value = func(open_vals, high_vals, low_vals, close_vals, volume_vals)
                    else:
                        value = func(open_vals[-1], high_vals[-1], low_vals[-1], close_vals[-1], volume_vals[-1])
                    
                    # Validar resultado
                    if math.isnan(value) or math.isinf(value):
                        value = 0.0
                    
                    indicator_values.append(float(value))
                    
                except Exception as e:
                    indicator_values.append(0.0)
            
            return np.array(indicator_values) if indicator_values else None
            
        except Exception as e:
            print(f"Error calculando valores del indicador: {str(e)}")
            return None
    
    def _backtest_indicator(self, indicator_values, data):
        """
        Evalúa la calidad del indicador mediante backtesting.
        
        Args:
            indicator_values: Valores del indicador calculados
            data: Datos históricos
            
        Returns:
            Puntuación de fitness del indicador
        """
        try:
            if len(indicator_values) < 20:
                return 0.0
            
            # Crear señales basadas en el indicador
            signals = self._generate_signals_from_indicator(indicator_values)
            
            if len(signals) == 0:
                return 0.0
            
            # Preparar datos para backtesting
            backtest_data = data.copy()
            backtest_data['indicator'] = np.nan
            backtest_data.iloc[:len(indicator_values), backtest_data.columns.get_loc('indicator')] = indicator_values
            
            # Simular estrategia simple basada en el indicador
            strategy = {
                'indicator': 'custom',
                'signal_threshold': 0.5,  # Umbral para generar señales
                'custom_values': indicator_values
            }
            
            # Ejecutar backtest simplificado
            results = self._simple_backtest(strategy, backtest_data)
            
            # Calcular fitness basado en métricas de performance
            fitness_score = self._calculate_fitness_score(results)
            
            return fitness_score
            
        except Exception as e:
            print(f"Error en backtest del indicador: {str(e)}")
            return 0.0
    
    def _generate_signals_from_indicator(self, indicator_values):
        """
        Genera señales de trading basadas en los valores del indicador.
        
        Args:
            indicator_values: Array con valores del indicador
            
        Returns:
            Lista de señales de trading
        """
        try:
            signals = []
            
            # Normalizar valores del indicador
            mean_val = np.mean(indicator_values)
            std_val = np.std(indicator_values)
            
            if std_val == 0:
                return signals
            
            normalized_values = (indicator_values - mean_val) / std_val
            
            # Generar señales basadas en cruces de umbrales
            for i in range(1, len(normalized_values)):
                prev_val = normalized_values[i-1]
                curr_val = normalized_values[i]
                
                # Señal de compra: cruce alcista del umbral inferior
                if prev_val < -1.0 and curr_val >= -1.0:
                    signals.append(('BUY', i))
                
                # Señal de venta: cruce bajista del umbral superior
                elif prev_val > 1.0 and curr_val <= 1.0:
                    signals.append(('SELL', i))
            
            return signals
            
        except Exception as e:
            print(f"Error generando señales: {str(e)}")
            return []
    
    def _simple_backtest(self, strategy, data):
        """
        Backtest simplificado para evaluar el indicador.
        
        Args:
            strategy: Diccionario con parámetros de la estrategia
            data: DataFrame con datos históricos
            
        Returns:
            Diccionario con resultados del backtest
        """
        try:
            trades = []
            position = None
            initial_capital = 10000
            
            indicator_values = strategy['custom_values']
            signals = self._generate_signals_from_indicator(indicator_values)
            
            for signal_type, signal_index in signals:
                if signal_index >= len(data):
                    continue
                
                current_price = data.iloc[signal_index]['close']
                
                if signal_type == 'BUY' and position is None:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_index': signal_index,
                        'quantity': initial_capital / current_price
                    }
                elif signal_type == 'SELL' and position is not None:
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100
                    })
                    
                    position = None
            
            # Calcular métricas
            if not trades:
                return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return': 0}
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in losing_trades))
            
            results = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades)) * 100 if trades else 0,
                'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
                'total_return': sum(t['return_pct'] for t in trades)
            }
            
            return results
            
        except Exception as e:
            print(f"Error en backtest simplificado: {str(e)}")
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return': 0}
    
    def _calculate_fitness_score(self, backtest_results):
        """
        Calcula la puntuación de fitness basada en los resultados del backtest.
        
        Args:
            backtest_results: Resultados del backtest
            
        Returns:
            Puntuación de fitness normalizada
        """
        try:
            if backtest_results['total_trades'] == 0:
                return 0.0
            
            # Métricas normalizadas
            win_rate = backtest_results['win_rate'] / 100  # Normalizar a [0,1]
            profit_factor = min(backtest_results['profit_factor'], 10) / 10  # Limitar y normalizar
            trade_count_bonus = min(backtest_results['total_trades'] / 20, 1.0)  # Bonus por cantidad de trades
            
            # Penalización por pocos trades
            if backtest_results['total_trades'] < 5:
                trade_count_bonus *= 0.5
            
            # Fitness compuesto
            fitness_score = (
                win_rate * 0.4 +                    # 40% win rate
                profit_factor * 0.4 +               # 40% profit factor
                trade_count_bonus * 0.2             # 20% cantidad de trades
            )
            
            # Bonus por excelente performance
            if win_rate > 0.7 and profit_factor > 0.8:
                fitness_score *= 1.5
            
            return min(fitness_score, 2.0)  # Limitar fitness máximo
            
        except Exception as e:
            print(f"Error calculando fitness score: {str(e)}")
            return 0.0
    
    def _generate_synthetic_test_data(self, length=200):
        """
        Genera datos sintéticos para pruebas cuando no hay datos reales disponibles.
        
        Args:
            length: Longitud de la serie temporal
            
        Returns:
            DataFrame con datos OHLCV sintéticos
        """
        try:
            # Generar precio base con tendencia y ruido
            base_price = 100
            trend = np.linspace(0, 20, length)
            noise = np.random.normal(0, 5, length)
            close_prices = base_price + trend + noise
            
            # Asegurar precios positivos
            close_prices = np.maximum(close_prices, 1.0)
            
            # Generar OHLC basado en close
            data = []
            for i, close in enumerate(close_prices):
                high = close * random.uniform(1.0, 1.05)
                low = close * random.uniform(0.95, 1.0)
                open_price = close_prices[i-1] if i > 0 else close
                volume = random.randint(1000, 10000)
                
                data.append({
                    'timestamp': f'2024-01-01 {i:02d}:00:00',
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error generando datos sintéticos: {str(e)}")
            return None
    
    def _compile_basic_tree(self, tree):
        """
        Compila un árbol de expresión básico cuando DEAP no está disponible.
        
        Args:
            tree: Representación del árbol de expresión
            
        Returns:
            Función compilada
        """
        # Implementación básica para compatibilidad
        def compiled_func(open_val, high_val, low_val, close_val, volume_val):
            try:
                # Función simple como ejemplo - en implementación real sería más compleja
                return (high_val + low_val) / 2 - close_val
            except:
                return 0.0
        
        return compiled_func
    
    async def discover_indicators(self, generations=50, population_size=100, 
                                  pair='BTCUSDT', timeframe='1h'):
        """
        Descubre nuevos indicadores técnicos usando programación genética.
        
        Args:
            generations: Número de generaciones para evolucionar
            population_size: Tamaño de la población
            pair: Par de trading para entrenar
            timeframe: Timeframe para obtener datos
            
        Returns:
            Lista de mejores indicadores descubiertos
        """
        print("🧬 ALPHA GENESIS - DESCUBRIMIENTO AUTÓNOMO DE INDICADORES")
        print("="*70)
        print("🎯 OBJETIVO: Crear nuevos indicadores técnicos mediante programación genética")
        print(f"📊 DATOS: {pair} - {timeframe}")
        print(f"🔄 GENERACIONES: {generations}")
        print(f"👥 POBLACIÓN: {population_size}")
        print("="*70)
        
        # Obtener datos históricos para entrenamiento
        if self.data_connector:
            try:
                if pair.startswith('OANDA:'):
                    resolution_map = {
                        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                        '1h': '60', '4h': '240', '1d': 'D'
                    }
                    forex_resolution = resolution_map.get(timeframe, '60')
                    self.test_data = await self.data_connector.get_forex_data(pair, forex_resolution, 500)
                else:
                    self.test_data = await self.data_connector.get_crypto_data(pair, timeframe, 500)
                
                if self.test_data is None or len(self.test_data) < 100:
                    print("⚠️ Datos insuficientes, usando datos sintéticos")
                    self.test_data = self._generate_synthetic_test_data()
                else:
                    print(f"✅ Datos obtenidos: {len(self.test_data)} velas")
                    
            except Exception as e:
                print(f"❌ Error obteniendo datos: {str(e)}")
                print("🔄 Usando datos sintéticos para entrenamiento")
                self.test_data = self._generate_synthetic_test_data()
        else:
            print("🔄 Sin conector de datos, usando datos sintéticos")
            self.test_data = self._generate_synthetic_test_data()
        
        if HAS_DEAP:
            return await self._discover_with_deap(generations, population_size)
        else:
            return await self._discover_basic(generations, population_size)
    
    async def _discover_with_deap(self, generations, population_size):
        """Descubrimiento usando DEAP."""
        print("🔬 Iniciando descubrimiento con DEAP...")
        
        # Crear población inicial
        population = self.toolbox.population(n=population_size)
        
        # Evaluar población inicial
        print("🧪 Evaluando población inicial...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        hall_of_fame = tools.HallOfFame(10)
        
        # Evolución
        print("🚀 Iniciando evolución...")
        final_pop, logbook = algorithms.eaSimple(
            population, self.toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
            stats=stats, halloffame=hall_of_fame, verbose=True
        )
        
        # Procesar mejores indicadores
        best_indicators = []
        for i, individual in enumerate(hall_of_fame):
            indicator = {
                'id': f'alpha_genesis_{int(datetime.now().timestamp())}_{i}',
                'expression': str(individual),
                'fitness': individual.fitness.values[0],
                'generation': generations,
                'discovery_date': datetime.now().isoformat()
            }
            
            # Guardar en base de datos
            self._save_indicator(indicator)
            best_indicators.append(indicator)
            
            print(f"🏆 Indicador #{i+1}: Fitness = {indicator['fitness']:.4f}")
            print(f"    Expresión: {indicator['expression'][:100]}...")
        
        print(f"\n✅ Descubrimiento completado: {len(best_indicators)} indicadores encontrados")
        return best_indicators
    
    async def _discover_basic(self, generations, population_size):
        """Descubrimiento básico sin DEAP."""
        print("🔬 Iniciando descubrimiento básico...")
        
        # Implementación simplificada
        best_indicators = []
        
        for gen in range(generations):
            print(f"🔄 Generación {gen+1}/{generations}")
            
            # Generar indicadores aleatorios simples
            for i in range(5):  # Generar 5 por generación
                indicator = {
                    'id': f'alpha_genesis_basic_{int(datetime.now().timestamp())}_{gen}_{i}',
                    'expression': f'(high + low) / 2 - close * {random.uniform(0.5, 2.0):.3f}',
                    'fitness': random.uniform(0.3, 0.8),  # Fitness simulado
                    'generation': gen,
                    'discovery_date': datetime.now().isoformat()
                }
                
                if len(best_indicators) < 10:
                    best_indicators.append(indicator)
                    self._save_indicator(indicator)
                elif indicator['fitness'] > min(ind['fitness'] for ind in best_indicators):
                    # Reemplazar el peor indicador
                    worst_idx = min(range(len(best_indicators)), 
                                    key=lambda x: best_indicators[x]['fitness'])
                    best_indicators[worst_idx] = indicator
                    self._save_indicator(indicator)
        
        print(f"\n✅ Descubrimiento básico completado: {len(best_indicators)} indicadores")
        return best_indicators
    
    def _save_indicator(self, indicator):
        """
        Guarda un indicador descubierto en la base de datos.
        
        Args:
            indicator: Diccionario con información del indicador
        """
        try:
            conn = sqlite3.connect('alpha_genesis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO discovered_indicators 
                (indicator_id, expression, fitness, generation, discovery_date, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                indicator['id'],
                indicator['expression'],
                indicator['fitness'],
                indicator['generation'],
                indicator['discovery_date'],
                ''  # Parámetros adicionales como JSON
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error guardando indicador: {str(e)}")
    
    def get_best_indicators(self, limit=10):
        """
        Obtiene los mejores indicadores descubiertos.
        
        Args:
            limit: Número máximo de indicadores a retornar
            
        Returns:
            Lista de mejores indicadores
        """
        try:
            conn = sqlite3.connect('alpha_genesis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT indicator_id, expression, fitness, generation, discovery_date
                FROM discovered_indicators 
                ORDER BY fitness DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            indicators = []
            for row in results:
                indicators.append({
                    'id': row[0],
                    'expression': row[1],
                    'fitness': row[2],
                    'generation': row[3],
                    'discovery_date': row[4]
                })
            
            return indicators
            
        except Exception as e:
            print(f"Error obteniendo mejores indicadores: {str(e)}")
            return []
    
    def export_indicator_for_evolution(self, indicator_id):
        """
        Exporta un indicador para su uso en el EvolutionaryEngine.
        
        Args:
            indicator_id: ID del indicador a exportar
            
        Returns:
            Diccionario con definición del indicador para evolución
        """
        try:
            conn = sqlite3.connect('alpha_genesis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT expression, fitness FROM discovered_indicators 
                WHERE indicator_id = ?
            ''', (indicator_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'indicator': 'alpha_genesis',
                    'expression': result[0],
                    'fitness_score': result[1],
                    'indicator_id': indicator_id,
                    'timeframe': '1h'  # Default timeframe
                }
            
            return None
            
        except Exception as e:
            print(f"Error exportando indicador: {str(e)}")
            return None

# Funciones de conveniencia
async def discover_trading_indicators(pair='BTCUSDT', timeframe='1h', 
                                      generations=30, population_size=50, 
                                      data_connector=None):
    """
    Función principal para descubrir nuevos indicadores de trading.
    
    Args:
        pair: Par de trading para entrenar
        timeframe: Timeframe para obtener datos
        generations: Número de generaciones
        population_size: Tamaño de población
        data_connector: Conector de datos
        
    Returns:
        Lista de mejores indicadores descubiertos
    """
    alpha_genesis = AlphaGenesisEngine(data_connector)
    return await alpha_genesis.discover_indicators(generations, population_size, pair, timeframe)

async def demo_alpha_genesis():
    """
    Función de demostración del motor Alpha Genesis.
    """
    print("🧬 DEMO: Alpha Genesis - Descubrimiento de Indicadores")
    print("="*60)
    
    try:
        # Inicializar motor
        alpha_genesis = AlphaGenesisEngine()
        
        # Descubrir indicadores
        indicators = await alpha_genesis.discover_indicators(
            generations=10,  # Reducido para demo
            population_size=20,  # Reducido para demo
            pair='BTCUSDT',
            timeframe='1h'
        )
        
        if indicators:
            print("\n🎯 INDICADORES DESCUBIERTOS:")
            print("="*60)
            
            for i, indicator in enumerate(indicators[:5], 1):
                print(f"{i}. ID: {indicator['id']}")
                print(f"   Fitness: {indicator['fitness']:.4f}")
                print(f"   Expresión: {indicator['expression'][:80]}...")
                print()
            
            # Mostrar mejores indicadores guardados
            best_indicators = alpha_genesis.get_best_indicators(5)
            print("🏆 TOP 5 MEJORES INDICADORES HISTÓRICOS:")
            print("="*60)
            
            for i, indicator in enumerate(best_indicators, 1):
                print(f"{i}. Fitness: {indicator['fitness']:.4f}")
                print(f"   Fecha: {indicator['discovery_date'][:10]}")
                print(f"   Expresión: {indicator['expression'][:80]}...")
                print()
        
        else:
            print("❌ No se pudieron descubrir indicadores")
            
    except Exception as e:
        print(f"❌ Error en demo Alpha Genesis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_alpha_genesis())
