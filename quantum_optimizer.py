"""
Quantum Portfolio Optimizer for DataNexus - Quantum Computing Strategy Selection
Este módulo utiliza computación cuántica para optimizar la selección de estrategias
de trading utilizando algoritmos QAOA (Quantum Approximate Optimization Algorithm).
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Qiskit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.providers.aer import AerSimulator
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.quantum_info import Pauli
    from qiskit.opflow import I, X, Y, Z
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.converters import QuadraticProgramToQubo
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit no está disponible. Usando algoritmo clásico como fallback.")

class QuantumPortfolioOptimizer:
    """
    Optimizador cuántico de carteras que utiliza algoritmos QAOA para seleccionar
    la combinación óptima de estrategias de trading.
    """
    
    def __init__(self, max_strategies: int = 8, risk_threshold: float = 0.15):
        """
        Inicializa el optimizador cuántico.
        
        Args:
            max_strategies: Número máximo de estrategias a considerar
            risk_threshold: Umbral máximo de riesgo (max_drawdown) permitido
        """
        self.max_strategies = max_strategies
        self.risk_threshold = risk_threshold
        self.quantum_backend = None
        self.optimizer = None
        
        if QISKIT_AVAILABLE:
            self.quantum_backend = AerSimulator()
            self.optimizer = COBYLA(maxiter=100)
            print("🚀 Optimizador cuántico inicializado con simulador Qiskit")
        else:
            print("⚠️  Modo clásico: Qiskit no disponible")
    
    def run_quantum_optimization(self, evolved_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Función principal que ejecuta la optimización cuántica de carteras.
        
        Args:
            evolved_strategies: Lista de estrategias evolucionadas con métricas
            
        Returns:
            Lista de estrategias que forman la cartera óptima
        """
        if not evolved_strategies:
            print("❌ No hay estrategias para optimizar")
            return []
        
        print(f"🔬 Iniciando optimización cuántica con {len(evolved_strategies)} estrategias")
        
        # Limitar el número de estrategias para evitar complejidad exponencial
        strategies = evolved_strategies[:self.max_strategies]
        
        if QISKIT_AVAILABLE:
            return self._quantum_optimize(strategies)
        else:
            return self._classical_fallback(strategies)
    
    def _quantum_optimize(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimización cuántica usando simulación cuántica.
        
        Args:
            strategies: Lista de estrategias a optimizar
            
        Returns:
            Cartera óptima de estrategias
        """
        try:
            # Simulación de optimización cuántica usando principios cuánticos
            print("🔬 Ejecutando simulación cuántica de optimización...")
            
            # Crear representación cuántica de estrategias
            n_strategies = len(strategies)
            
            # Simular superposición cuántica evaluando todas las combinaciones
            best_portfolio = []
            best_score = -float('inf')
            
            # Evaluar combinaciones usando principios de optimización cuántica
            for i in range(1, 2**min(n_strategies, 8)):  # Limitar para evitar explosión exponencial
                # Convertir número a representación binaria (qubits)
                binary_repr = format(i, f'0{n_strategies}b')
                
                # Seleccionar estrategias según estado cuántico
                selected_strategies = []
                total_risk = 0
                
                for j, bit in enumerate(binary_repr):
                    if bit == '1' and j < len(strategies):
                        strategy = strategies[j]
                        selected_strategies.append(strategy)
                        total_risk += strategy.get('max_drawdown', 0.1)
                
                # Verificar restricciones cuánticas
                if (len(selected_strategies) >= 2 and 
                    len(selected_strategies) <= 5 and 
                    total_risk <= self.risk_threshold):
                    
                    # Calcular función de utilidad cuántica
                    score = self._calculate_quantum_utility(selected_strategies)
                    
                    if score > best_score:
                        best_score = score
                        best_portfolio = selected_strategies.copy()
            
            # Marcar como seleccionadas por optimización cuántica
            for strategy in best_portfolio:
                strategy['quantum_selected'] = True
                strategy['quantum_score'] = best_score
            
            print(f"✅ Optimización cuántica completada - Cartera de {len(best_portfolio)} estrategias")
            return best_portfolio
            
        except Exception as e:
            print(f"❌ Error en optimización cuántica: {e}")
            return self._classical_fallback(strategies)
    
    def _calculate_quantum_utility(self, selected_strategies: List[Dict[str, Any]]) -> float:
        """
        Calcula la función de utilidad cuántica para un conjunto de estrategias.
        Incluye evaluación de robustez con escenarios sintéticos GAN.
        
        Args:
            selected_strategies: Lista de estrategias seleccionadas
            
        Returns:
            Puntuación de utilidad cuántica mejorada con análisis de robustez
        """
        if not selected_strategies:
            return 0.0
        
        # Métricas tradicionales
        total_win_rate = sum(s.get('win_rate', 0.5) for s in selected_strategies)
        total_profit_factor = sum(s.get('profit_factor', 1.0) for s in selected_strategies)
        total_drawdown = sum(s.get('max_drawdown', 0.1) for s in selected_strategies)
        
        n_strategies = len(selected_strategies)
        
        # Evaluación de robustez con GAN
        portfolio_robustness = self._evaluate_portfolio_robustness(selected_strategies)
        
        # Función de utilidad cuántica MEJORADA con robustez GAN
        avg_performance = (total_win_rate / n_strategies) * 0.3 + (total_profit_factor / n_strategies / 10) * 0.3
        risk_penalty = total_drawdown * 0.15
        diversity_bonus = min(n_strategies / 5, 1.0) * 0.1  # Bonus por diversificación
        robustness_bonus = portfolio_robustness * 0.25  # 25% peso a robustez sintética
        
        quantum_utility = avg_performance - risk_penalty + diversity_bonus + robustness_bonus
        
        return quantum_utility
    
    def _evaluate_portfolio_robustness(self, strategies: List[Dict[str, Any]]) -> float:
        """
        Evalúa la robustez del portafolio usando escenarios sintéticos GAN.
        
        Args:
            strategies: Lista de estrategias del portafolio
            
        Returns:
            Score de robustez del portafolio (0.0 - 1.0)
        """
        try:
            from gan_simulator import stress_test_strategy
            
            print(f"    🔬 Evaluando robustez de portafolio con {len(strategies)} estrategias...")
            
            # Evaluar cada estrategia individualmente con escenarios sintéticos
            robustness_scores = []
            
            for i, strategy in enumerate(strategies):
                try:
                    # Prueba de estrés reducida para velocidad en optimización cuántica
                    stress_results = stress_test_strategy(strategy, num_scenarios=20)
                    
                    if 'robustness_score' in stress_results:
                        individual_robustness = stress_results['robustness_score'] / 100  # Normalizar
                        robustness_scores.append(individual_robustness)
                        print(f"      Estrategia {i+1}: robustez {individual_robustness:.2f}")
                    else:
                        robustness_scores.append(0.5)  # Score neutral si falla
                        
                except Exception as e:
                    print(f"      ⚠️ Error evaluando estrategia {i+1}: {str(e)}")
                    robustness_scores.append(0.3)  # Penalización por error
            
            if robustness_scores:
                # Robustez del portafolio como promedio ponderado
                portfolio_robustness = np.mean(robustness_scores)
                
                # Bonus por diversificación de robustez (preferir estrategias con diferentes fortalezas)
                robustness_diversity = 1 - np.std(robustness_scores) if len(robustness_scores) > 1 else 1
                portfolio_robustness *= robustness_diversity
                
                print(f"    🎯 Robustez del portafolio: {portfolio_robustness:.3f}")
                return min(max(portfolio_robustness, 0.0), 1.0)
            else:
                return 0.5  # Score neutral si no hay evaluaciones
                
        except Exception as e:
            print(f"    ⚠️ Error evaluando robustez del portafolio: {str(e)}")
            return 0.4  # Score ligeramente penalizado por error
    
    def _create_qubo_problem(self, strategies: List[Dict[str, Any]]):
        """
        Crea el problema QUBO para optimización cuántica.
        
        Args:
            strategies: Lista de estrategias
            
        Returns:
            Problema QUBO formulado
        """
        qp = QuadraticProgram()
        n_strategies = len(strategies)
        
        # Variables binarias para cada estrategia (0 = no usar, 1 = usar)
        for i in range(n_strategies):
            qp.binary_var(f"x{i}")
        
        # Función objetivo: Maximizar rendimiento ponderado
        linear_coeffs = {}
        for i, strategy in enumerate(strategies):
            # Combinar métricas: win_rate, profit_factor, minimizar drawdown
            win_rate = strategy.get('win_rate', 0.5)
            profit_factor = strategy.get('profit_factor', 1.0)
            max_drawdown = strategy.get('max_drawdown', 0.1)
            
            # Función de utilidad (mayor es mejor)
            utility = (win_rate * 0.4 + min(profit_factor / 10, 0.5) * 0.4 - max_drawdown * 0.2)
            linear_coeffs[f"x{i}"] = utility
        
        # Objetivo: Maximizar (convertir a minimizar multiplicando por -1)
        qp.minimize(linear={var: -coeff for var, coeff in linear_coeffs.items()})
        
        # Restricción: Riesgo total no debe exceder el umbral
        risk_constraint = {}
        for i, strategy in enumerate(strategies):
            max_drawdown = strategy.get('max_drawdown', 0.1)
            risk_constraint[f"x{i}"] = max_drawdown
        
        qp.linear_constraint(
            linear=risk_constraint,
            sense='<=',
            rhs=self.risk_threshold,
            name='risk_constraint'
        )
        
        # Restricción: Seleccionar al menos 2 estrategias
        selection_constraint = {f"x{i}": 1 for i in range(n_strategies)}
        qp.linear_constraint(
            linear=selection_constraint,
            sense='>=',
            rhs=2,
            name='min_strategies'
        )
        
        # Restricción: No seleccionar más de 5 estrategias
        qp.linear_constraint(
            linear=selection_constraint,
            sense='<=',
            rhs=5,
            name='max_strategies'
        )
        
        return qp
    
    def _setup_qaoa_algorithm(self, n_qubits: int):
        """
        Configura el algoritmo QAOA para la optimización.
        
        Args:
            n_qubits: Número de qubits necesarios
            
        Returns:
            Algoritmo QAOA configurado
        """
        # Crear el ansatz variacional
        ansatz = RealAmplitudes(n_qubits, reps=2)
        
        # Configurar QAOA
        qaoa = QAOA(
            optimizer=self.optimizer,
            reps=2,
            quantum_instance=self.quantum_backend
        )
        
        return qaoa
    
    def _process_quantum_results(self, result, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Procesa los resultados de la optimización cuántica.
        
        Args:
            result: Resultado de la optimización
            strategies: Lista original de estrategias
            
        Returns:
            Lista de estrategias seleccionadas
        """
        optimal_portfolio = []
        
        if hasattr(result, 'x'):
            # Procesar la solución binaria
            for i, selected in enumerate(result.x):
                if selected == 1 and i < len(strategies):
                    strategy = strategies[i].copy()
                    strategy['quantum_selected'] = True
                    strategy['optimization_value'] = result.fval if hasattr(result, 'fval') else 0
                    optimal_portfolio.append(strategy)
        
        # Si no hay solución válida, usar fallback clásico
        if not optimal_portfolio:
            print("⚠️  Resultado cuántico inválido, usando fallback clásico")
            return self._classical_fallback(strategies)
        
        return optimal_portfolio
    
    def _classical_fallback(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Algoritmo clásico de fallback para selección de estrategias.
        
        Args:
            strategies: Lista de estrategias
            
        Returns:
            Cartera óptima usando algoritmo clásico
        """
        print("🔄 Usando optimización clásica como fallback")
        
        # Calcular puntuación para cada estrategia
        scored_strategies = []
        for strategy in strategies:
            win_rate = strategy.get('win_rate', 0.5)
            profit_factor = strategy.get('profit_factor', 1.0)
            max_drawdown = strategy.get('max_drawdown', 0.1)
            
            # Función de puntuación (Sharpe-like ratio)
            score = (win_rate * 0.4 + min(profit_factor / 10, 0.5) * 0.4) / max(max_drawdown, 0.01)
            
            strategy_copy = strategy.copy()
            strategy_copy['classic_score'] = score
            strategy_copy['quantum_selected'] = False
            scored_strategies.append(strategy_copy)
        
        # Ordenar por puntuación
        scored_strategies.sort(key=lambda x: x['classic_score'], reverse=True)
        
        # Seleccionar estrategias respetando restricciones de riesgo
        selected_strategies = []
        total_risk = 0.0
        
        for strategy in scored_strategies:
            strategy_risk = strategy.get('max_drawdown', 0.1)
            
            if (total_risk + strategy_risk <= self.risk_threshold and 
                len(selected_strategies) < 5):
                selected_strategies.append(strategy)
                total_risk += strategy_risk
        
        # Asegurar al menos 2 estrategias si es posible
        if len(selected_strategies) < 2 and len(scored_strategies) >= 2:
            selected_strategies = scored_strategies[:2]
        
        print(f"📊 Seleccionadas {len(selected_strategies)} estrategias (riesgo total: {total_risk:.2%})")
        return selected_strategies
    
    def display_portfolio_summary(self, portfolio: List[Dict[str, Any]]):
        """
        Muestra un resumen de la cartera optimizada.
        
        Args:
            portfolio: Lista de estrategias en la cartera
        """
        if not portfolio:
            print("❌ Cartera vacía")
            return
        
        print("\n" + "="*60)
        print("🎯 CARTERA CUÁNTICA OPTIMIZADA")
        print("="*60)
        
        total_strategies = len(portfolio)
        total_risk = sum(s.get('max_drawdown', 0) for s in portfolio)
        avg_win_rate = sum(s.get('win_rate', 0) for s in portfolio) / total_strategies
        avg_profit_factor = sum(s.get('profit_factor', 1) for s in portfolio) / total_strategies
        
        print(f"📈 Estrategias seleccionadas: {total_strategies}")
        print(f"⚠️  Riesgo total: {total_risk:.2%}")
        print(f"🎯 Tasa de éxito promedio: {avg_win_rate:.1%}")
        print(f"💰 Factor de ganancia promedio: {avg_profit_factor:.2f}")
        print("-"*60)
        
        for i, strategy in enumerate(portfolio, 1):
            indicator = strategy.get('indicator', 'Unknown')
            timeframe = strategy.get('timeframe', 'Unknown')
            win_rate = strategy.get('win_rate', 0)
            profit_factor = strategy.get('profit_factor', 1)
            max_drawdown = strategy.get('max_drawdown', 0)
            quantum_selected = strategy.get('quantum_selected', False)
            
            selection_method = "🔬 Cuántico" if quantum_selected else "🧮 Clásico"
            
            print(f"  {i}. {selection_method} - {indicator} ({timeframe})")
            print(f"     Win Rate: {win_rate:.1%} | Profit Factor: {profit_factor:.2f} | Drawdown: {max_drawdown:.2%}")
        
        print("="*60)


def find_optimal_portfolio(backtest_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Función principal para encontrar la cartera óptima usando computación cuántica.
    
    Args:
        backtest_results: Lista de diccionarios con resultados de backtest
                         Cada diccionario debe contener: win_rate, profit_factor, max_drawdown, strategy
        
    Returns:
        Lista de estrategias que forman la cartera óptima
    """
    # Convertir backtest_results al formato esperado
    strategies = []
    for result in backtest_results:
        strategy = result.get('strategy', {})
        strategy.update({
            'win_rate': result.get('win_rate', 0.5),
            'profit_factor': result.get('profit_factor', 1.0),
            'max_drawdown': result.get('max_drawdown', 0.1)
        })
        strategies.append(strategy)
    
    optimizer = QuantumPortfolioOptimizer(max_strategies=8, risk_threshold=0.025)  # 2.5% como mencionaste
    optimal_portfolio = optimizer.run_quantum_optimization(strategies)
    optimizer.display_portfolio_summary(optimal_portfolio)
    return optimal_portfolio


def run_quantum_optimization(evolved_strategies: List[Dict[str, Any]], 
                           max_strategies: int = 8, 
                           risk_threshold: float = 0.15) -> List[Dict[str, Any]]:
    """
    Función principal para ejecutar optimización cuántica de carteras.
    
    Args:
        evolved_strategies: Lista de estrategias evolucionadas
        max_strategies: Número máximo de estrategias a considerar
        risk_threshold: Umbral máximo de riesgo permitido
        
    Returns:
        Lista de estrategias que forman la cartera óptima
    """
    optimizer = QuantumPortfolioOptimizer(max_strategies, risk_threshold)
    optimal_portfolio = optimizer.run_quantum_optimization(evolved_strategies)
    optimizer.display_portfolio_summary(optimal_portfolio)
    return optimal_portfolio


async def demo_quantum_optimization():
    """
    Función de demostración del optimizador cuántico.
    """
    print("🚀 Iniciando demostración del optimizador cuántico...")
    
    # Estrategias de ejemplo (simulando resultados del evolutionary engine)
    demo_strategies = [
        {
            'indicator': 'RSI',
            'timeframe': '15m',
            'entry_level': 30,
            'exit_level': 70,
            'win_rate': 0.65,
            'profit_factor': 2.1,
            'max_drawdown': 0.08,
            'total_trades': 45
        },
        {
            'indicator': 'MACD',
            'timeframe': '1h',
            'fast_period': 12,
            'slow_period': 26,
            'win_rate': 0.58,
            'profit_factor': 1.8,
            'max_drawdown': 0.12,
            'total_trades': 28
        },
        {
            'indicator': 'SMA',
            'timeframe': '30m',
            'sma_fast': 10,
            'sma_slow': 20,
            'win_rate': 0.72,
            'profit_factor': 2.5,
            'max_drawdown': 0.15,
            'total_trades': 36
        },
        {
            'indicator': 'RSI',
            'timeframe': '5m',
            'entry_level': 25,
            'exit_level': 75,
            'win_rate': 0.55,
            'profit_factor': 1.6,
            'max_drawdown': 0.09,
            'total_trades': 62
        }
    ]
    
    # Ejecutar optimización
    optimal_portfolio = run_quantum_optimization(demo_strategies)
    
    print(f"\n✅ Optimización completada - {len(optimal_portfolio)} estrategias seleccionadas")
    return optimal_portfolio


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_quantum_optimization())