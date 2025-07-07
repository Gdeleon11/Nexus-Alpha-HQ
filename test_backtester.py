#!/usr/bin/env python3
"""
Test script for the backtester module.
Demonstrates different trading strategies and their performance.
"""

import asyncio
import pandas as pd
from backtester import run_backtest, BacktesterEngine
from datanexus_connector import DataNexus

async def test_rsi_strategy():
    """
    Test RSI-based trading strategy.
    """
    print("🎯 PRUEBA DE ESTRATEGIA RSI")
    print("="*50)
    
    # Obtener datos históricos
    data_nexus = DataNexus()
    print("📊 Obteniendo datos históricos de BTC-USDT...")
    historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 200)
    
    if historical_data is None:
        print("❌ Error: No se pudieron obtener datos históricos")
        return
    
    print(f"✅ Datos obtenidos: {len(historical_data)} velas")
    
    # Estrategia RSI conservadora
    strategy = {
        'indicator': 'RSI',
        'entry_level': 30,    # Comprar cuando RSI < 30 (sobreventa)
        'exit_level': 70,     # Vender cuando RSI > 70 (sobrecompra)
        'rsi_period': 14
    }
    
    print(f"🔧 Estrategia: {strategy}")
    
    # Ejecutar backtest
    results = run_backtest(strategy, historical_data)
    
    # Mostrar resultados
    print_results(results, "RSI Conservadora")
    
    return results

async def test_aggressive_rsi_strategy():
    """
    Test aggressive RSI strategy.
    """
    print("\n🎯 PRUEBA DE ESTRATEGIA RSI AGRESIVA")
    print("="*50)
    
    # Obtener datos históricos
    data_nexus = DataNexus()
    historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 200)
    
    if historical_data is None:
        print("❌ Error: No se pudieron obtener datos históricos")
        return
    
    # Estrategia RSI agresiva
    strategy = {
        'indicator': 'RSI',
        'entry_level': 40,    # Entrada más temprana
        'exit_level': 60,     # Salida más temprana
        'rsi_period': 14
    }
    
    print(f"🔧 Estrategia: {strategy}")
    
    # Ejecutar backtest
    results = run_backtest(strategy, historical_data)
    
    # Mostrar resultados
    print_results(results, "RSI Agresiva")
    
    return results

async def test_sma_strategy():
    """
    Test SMA crossover strategy.
    """
    print("\n🎯 PRUEBA DE ESTRATEGIA SMA CROSSOVER")
    print("="*50)
    
    # Obtener datos históricos
    data_nexus = DataNexus()
    historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 200)
    
    if historical_data is None:
        print("❌ Error: No se pudieron obtener datos históricos")
        return
    
    # Estrategia SMA crossover
    strategy = {
        'indicator': 'SMA',
        'sma_fast': 10,       # SMA rápida de 10 períodos
        'sma_slow': 20        # SMA lenta de 20 períodos
    }
    
    print(f"🔧 Estrategia: {strategy}")
    
    # Ejecutar backtest
    results = run_backtest(strategy, historical_data)
    
    # Mostrar resultados
    print_results(results, "SMA Crossover")
    
    return results

async def test_macd_strategy():
    """
    Test MACD strategy.
    """
    print("\n🎯 PRUEBA DE ESTRATEGIA MACD")
    print("="*50)
    
    # Obtener datos históricos
    data_nexus = DataNexus()
    historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 200)
    
    if historical_data is None:
        print("❌ Error: No se pudieron obtener datos históricos")
        return
    
    # Estrategia MACD
    strategy = {
        'indicator': 'MACD'
    }
    
    print(f"🔧 Estrategia: {strategy}")
    
    # Ejecutar backtest
    results = run_backtest(strategy, historical_data)
    
    # Mostrar resultados
    print_results(results, "MACD")
    
    return results

def print_results(results: dict, strategy_name: str):
    """
    Print formatted backtest results.
    
    Args:
        results: Dictionary with backtest results
        strategy_name: Name of the strategy
    """
    print(f"\n📊 RESULTADOS - {strategy_name.upper()}")
    print("="*50)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"📈 Operaciones totales: {results['total_trades']}")
    print(f"🏆 Operaciones ganadoras: {results['winning_trades']}")
    print(f"❌ Operaciones perdedoras: {results['losing_trades']}")
    print(f"🎯 Tasa de éxito: {results['win_rate']:.2f}%")
    print(f"💰 Factor de ganancia: {results['profit_factor']:.2f}")
    print(f"📉 Drawdown máximo: {results['max_drawdown']:.2f}%")
    print(f"💵 Retorno total: {results['total_return']:.2f}%")
    print(f"💲 Valor final: ${results['final_value']:.2f}")
    
    if results['total_trades'] > 0:
        print(f"💰 Ganancia total: ${results['total_profit']:.2f}")
        print(f"💸 Pérdida total: ${results['total_loss']:.2f}")
        
        # Mostrar últimas 3 operaciones
        if results['trades_detail']:
            print("\n📋 Últimas operaciones:")
            for i, trade in enumerate(results['trades_detail'][-3:], 1):
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                print(f"   {i}. {pnl_emoji} {trade['direction']} - P&L: ${trade['pnl']:.2f} ({trade['return_pct']:.2f}%)")
    
    print("="*50)

async def compare_strategies():
    """
    Compare multiple strategies side by side.
    """
    print("\n🏆 COMPARACIÓN DE ESTRATEGIAS")
    print("="*70)
    
    # Ejecutar todas las estrategias
    strategies = []
    
    # RSI Conservadora
    rsi_conservative = await test_rsi_strategy()
    strategies.append(("RSI Conservadora", rsi_conservative))
    
    # RSI Agresiva
    rsi_aggressive = await test_aggressive_rsi_strategy()
    strategies.append(("RSI Agresiva", rsi_aggressive))
    
    # SMA Crossover
    sma_results = await test_sma_strategy()
    strategies.append(("SMA Crossover", sma_results))
    
    # MACD
    macd_results = await test_macd_strategy()
    strategies.append(("MACD", macd_results))
    
    # Tabla comparativa
    print("\n📊 TABLA COMPARATIVA")
    print("="*70)
    print(f"{'Estrategia':<15} {'Trades':<8} {'Win %':<8} {'Profit Factor':<12} {'Return %':<10} {'Max DD %':<10}")
    print("-" * 70)
    
    for name, results in strategies:
        if 'error' not in results:
            print(f"{name:<15} {results['total_trades']:<8} {results['win_rate']:<8.1f} {results['profit_factor']:<12.2f} {results['total_return']:<10.2f} {results['max_drawdown']:<10.2f}")
    
    print("="*70)
    
    # Mejor estrategia
    best_strategy = max(strategies, key=lambda x: x[1].get('total_return', -float('inf')))
    print(f"🏆 MEJOR ESTRATEGIA: {best_strategy[0]}")
    print(f"💰 Retorno: {best_strategy[1]['total_return']:.2f}%")
    print("="*70)

async def main():
    """
    Main function to run all backtesting tests.
    """
    print("🚀 INICIANDO PRUEBAS DE BACKTESTING")
    print("="*70)
    print("📊 Sistema de simulación de estrategias de trading")
    print("🎯 Probando múltiples estrategias con datos reales")
    print("="*70)
    
    try:
        await compare_strategies()
        
        print("\n✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("🎉 El sistema de backtesting está funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error durante las pruebas: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())