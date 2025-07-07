#!/usr/bin/env python3
"""
Simple test for the backtester module without pandas_ta dependency issues.
"""

import asyncio
import pandas as pd
from backtester import run_backtest
from datanexus_connector import DataNexus

async def test_simple_backtest():
    """Test basic backtesting functionality."""
    print("🎯 PRUEBA SIMPLE DE BACKTESTING")
    print("="*50)
    
    try:
        # Initialize DataNexus
        data_nexus = DataNexus()
        
        # Get historical data
        print("📊 Obteniendo datos históricos de BTC-USDT...")
        historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 100)
        
        if historical_data is None:
            print("❌ Error: No se pudieron obtener datos históricos")
            return
        
        print(f"✅ Datos obtenidos: {len(historical_data)} velas")
        print(f"📈 Rango de precios: ${historical_data['close'].min():.2f} - ${historical_data['close'].max():.2f}")
        
        # Simple RSI strategy
        strategy = {
            'indicator': 'RSI',
            'entry_level': 30,
            'exit_level': 70,
            'rsi_period': 14
        }
        
        print(f"🔧 Estrategia RSI: Comprar cuando RSI < {strategy['entry_level']}, Vender cuando RSI > {strategy['exit_level']}")
        
        # Run backtest
        print("\n🚀 Ejecutando backtest...")
        results = run_backtest(strategy, historical_data)
        
        # Display results
        print("\n📊 RESULTADOS:")
        print("="*50)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📈 Operaciones totales: {results['total_trades']}")
            print(f"🏆 Operaciones ganadoras: {results['winning_trades']}")
            print(f"❌ Operaciones perdedoras: {results['losing_trades']}")
            print(f"🎯 Tasa de éxito: {results['win_rate']:.2f}%")
            print(f"💰 Factor de ganancia: {results['profit_factor']:.2f}")
            print(f"📉 Drawdown máximo: {results['max_drawdown']:.2f}%")
            print(f"💵 Retorno total: {results['total_return']:.2f}%")
            print(f"💲 Valor inicial: $10,000.00")
            print(f"💲 Valor final: ${results['final_value']:.2f}")
            
            if results['total_trades'] > 0:
                print(f"\n📋 Detalles de operaciones:")
                for i, trade in enumerate(results['trades_detail'][:3], 1):
                    pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                    print(f"   {i}. {pnl_emoji} {trade['direction']} - Entrada: ${trade['entry_price']:.2f}, Salida: ${trade['exit_price']:.2f}, P&L: ${trade['pnl']:.2f}")
        
        print("="*50)
        print("✅ Prueba de backtesting completada!")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_backtest())