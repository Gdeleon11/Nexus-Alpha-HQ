#!/usr/bin/env python3
"""
Test script for the evolutionary engine.
Demonstrates genetic algorithm optimization of trading strategies.
"""

import asyncio
from evolutionary_engine import evolve_trading_strategy, EvolutionaryEngine
from datanexus_connector import DataNexus
from backtester import run_backtest

async def test_evolution_quick():
    """
    Test rápido del motor evolutivo con parámetros reducidos.
    """
    print("🧬 PRUEBA RÁPIDA DEL MOTOR EVOLUTIVO")
    print("="*50)
    
    try:
        # Evolucionar estrategia con parámetros pequeños para prueba rápida
        print("🚀 Iniciando evolución (prueba rápida)...")
        best_strategy = await evolve_trading_strategy(
            pair='BTCUSDT',
            generations=5,  # Solo 5 generaciones para prueba rápida
            population_size=20  # Población pequeña
        )
        
        if best_strategy:
            print("\n✅ EVOLUCIÓN EXITOSA")
            print("="*50)
            print(f"🏆 Estrategia evolucionada: {best_strategy}")
            
            # Probar la estrategia
            print("\n🧪 Probando estrategia evolucionada...")
            data_nexus = DataNexus()
            test_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 50)
            
            if test_data is not None:
                results = run_backtest(best_strategy, test_data)
                
                print("📊 RESULTADOS DEL BACKTEST:")
                print(f"   📈 Operaciones totales: {results['total_trades']}")
                print(f"   🎯 Tasa de éxito: {results['win_rate']:.2f}%")
                print(f"   💰 Factor de ganancia: {results['profit_factor']:.2f}")
                print(f"   💵 Retorno total: {results['total_return']:.2f}%")
                print(f"   📉 Drawdown máximo: {results['max_drawdown']:.2f}%")
                
                return best_strategy
            else:
                print("❌ No se pudieron obtener datos para la prueba")
        else:
            print("❌ No se pudo evolucionar una estrategia válida")
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

async def test_evolution_comparison():
    """
    Compara estrategia evolucionada vs estrategia fija.
    """
    print("\n🏆 COMPARACIÓN: EVOLUCIÓN VS ESTRATEGIA FIJA")
    print("="*60)
    
    try:
        # Obtener datos para comparación
        data_nexus = DataNexus()
        test_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 100)
        
        if test_data is None:
            print("❌ No se pudieron obtener datos para comparación")
            return
        
        # Estrategia fija (RSI tradicional)
        fixed_strategy = {
            'indicator': 'RSI',
            'entry_level': 30,
            'exit_level': 70,
            'rsi_period': 14
        }
        
        print("🧪 Probando estrategia fija (RSI tradicional)...")
        fixed_results = run_backtest(fixed_strategy, test_data)
        
        # Evolucionar estrategia
        print("🧬 Evolucionando estrategia optimizada...")
        evolved_strategy = await evolve_trading_strategy(
            pair='BTCUSDT',
            generations=8,
            population_size=25
        )
        
        if evolved_strategy:
            print("🧪 Probando estrategia evolucionada...")
            evolved_results = run_backtest(evolved_strategy, test_data)
            
            # Comparación
            print("\n📊 COMPARACIÓN DE RESULTADOS:")
            print("="*60)
            print(f"{'Métrica':<20} {'Fija':<15} {'Evolucionada':<15} {'Mejora':<10}")
            print("-" * 60)
            
            metrics = [
                ('Operaciones', 'total_trades', ''),
                ('Tasa de éxito (%)', 'win_rate', '%'),
                ('Factor ganancia', 'profit_factor', ''),
                ('Retorno total (%)', 'total_return', '%'),
                ('Max drawdown (%)', 'max_drawdown', '%')
            ]
            
            for name, key, unit in metrics:
                fixed_val = fixed_results.get(key, 0)
                evolved_val = evolved_results.get(key, 0)
                
                if key == 'max_drawdown':
                    improvement = "Mejor" if evolved_val < fixed_val else "Peor"
                else:
                    improvement = "Mejor" if evolved_val > fixed_val else "Peor"
                
                print(f"{name:<20} {fixed_val:<15.2f} {evolved_val:<15.2f} {improvement:<10}")
            
            print("="*60)
            
            # Mostrar estrategias
            print(f"\n🔧 ESTRATEGIA FIJA: {fixed_strategy}")
            print(f"🧬 ESTRATEGIA EVOLUCIONADA: {evolved_strategy}")
            
        else:
            print("❌ No se pudo evolucionar estrategia para comparación")
            
    except Exception as e:
        print(f"❌ Error durante la comparación: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """
    Función principal para ejecutar todas las pruebas del motor evolutivo.
    """
    print("🧬 INICIANDO PRUEBAS DEL MOTOR EVOLUTIVO")
    print("="*60)
    print("🎯 Optimización de estrategias de trading con algoritmos genéticos")
    print("="*60)
    
    # Prueba rápida
    evolved_strategy = await test_evolution_quick()
    
    if evolved_strategy:
        # Comparación si la evolución fue exitosa
        await test_evolution_comparison()
        
        print("\n✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("🎉 El motor evolutivo está funcionando correctamente!")
    else:
        print("\n❌ Las pruebas no se completaron exitosamente")

if __name__ == "__main__":
    asyncio.run(main())