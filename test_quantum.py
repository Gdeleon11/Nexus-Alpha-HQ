#!/usr/bin/env python3
"""
Test script for the Quantum Portfolio Optimizer
Este script prueba la función find_optimal_portfolio con datos de backtest simulados.
"""

from quantum_optimizer import find_optimal_portfolio

def main():
    """
    Script principal de prueba para el optimizador cuántico.
    """
    print("🚀 Testing Quantum Portfolio Optimizer...")
    
    # Crear datos de backtest de prueba (dummy data) para 8 estrategias
    backtest_results = [
        {
            'strategy': {
                'indicator': 'RSI',
                'timeframe': '15m',
                'entry_level': 30,
                'exit_level': 70,
                'rsi_period': 14
            },
            'win_rate': 0.65,
            'profit_factor': 2.10,
            'max_drawdown': 0.08,
            'total_trades': 45
        },
        {
            'strategy': {
                'indicator': 'RSI',
                'timeframe': '5m',
                'entry_level': 25,
                'exit_level': 75,
                'rsi_period': 12
            },
            'win_rate': 0.55,
            'profit_factor': 1.60,
            'max_drawdown': 0.09,
            'total_trades': 78
        },
        {
            'strategy': {
                'indicator': 'MACD',
                'timeframe': '1h',
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'win_rate': 0.58,
            'profit_factor': 1.85,
            'max_drawdown': 0.12,
            'total_trades': 32
        },
        {
            'strategy': {
                'indicator': 'MACD',
                'timeframe': '30m',
                'fast_period': 8,
                'slow_period': 21,
                'signal_period': 7
            },
            'win_rate': 0.62,
            'profit_factor': 1.95,
            'max_drawdown': 0.11,
            'total_trades': 28
        },
        {
            'strategy': {
                'indicator': 'SMA',
                'timeframe': '1h',
                'sma_fast': 10,
                'sma_slow': 20
            },
            'win_rate': 0.52,
            'profit_factor': 1.40,
            'max_drawdown': 0.15,
            'total_trades': 22
        },
        {
            'strategy': {
                'indicator': 'SMA',
                'timeframe': '15m',
                'sma_fast': 5,
                'sma_slow': 15
            },
            'win_rate': 0.68,
            'profit_factor': 2.25,
            'max_drawdown': 0.14,
            'total_trades': 56
        },
        {
            'strategy': {
                'indicator': 'RSI',
                'timeframe': '4h',
                'entry_level': 35,
                'exit_level': 65,
                'rsi_period': 16
            },
            'win_rate': 0.70,
            'profit_factor': 2.40,
            'max_drawdown': 0.16,
            'total_trades': 18
        },
        {
            'strategy': {
                'indicator': 'MACD',
                'timeframe': '15m',
                'fast_period': 6,
                'slow_period': 18,
                'signal_period': 5
            },
            'win_rate': 0.48,
            'profit_factor': 1.20,
            'max_drawdown': 0.18,
            'total_trades': 95
        }
    ]
    
    print(f"📊 Analizando {len(backtest_results)} estrategias...")
    
    # Llamar a la función de optimización cuántica
    optimal_portfolio = find_optimal_portfolio(backtest_results)
    
    # Imprimir resultados finales
    print(f"\n✅ Optimización completada - {len(optimal_portfolio)} estrategias seleccionadas")
    if optimal_portfolio:
        print("🏆 Cartera Óptima:")
        for i, strategy in enumerate(optimal_portfolio, 1):
            indicator = strategy.get('indicator', 'Unknown')
            timeframe = strategy.get('timeframe', 'Unknown')
            win_rate = strategy.get('win_rate', 0)
            print(f"  {i}. {indicator} ({timeframe}) - Win Rate: {win_rate:.1%}")
    else:
        print("❌ No se pudo optimizar la cartera")

if __name__ == "__main__":
    main()