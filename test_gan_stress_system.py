
#!/usr/bin/env python3
"""
Test completo del sistema de stress testing con GAN integrado.
Demuestra cómo el motor evolutivo usa escenarios sintéticos para evaluar estrategias.
"""

import asyncio
import pandas as pd
from typing import Dict, Any

from datanexus_connector import DataNexus
from gan_simulator import GANMarketSimulator, train_market_gan, generate_scenarios, stress_test_strategy
from evolutionary_engine import evolve_trading_strategy, EvolutionaryEngine
from backtester import run_backtest

async def demo_complete_gan_integration():
    """
    Demostración completa del sistema integrado GAN + Motor Evolutivo.
    """
    print("🚀 DEMO COMPLETO: SISTEMA DE STRESS TESTING CON GAN SINTÉTICO")
    print("="*80)
    print("🎯 FLUJO COMPLETO:")
    print("   1. Obtener datos históricos para entrenamiento")
    print("   2. Entrenar modelo TimeGAN para generar mercados sintéticos")
    print("   3. Evolucionar estrategias con stress testing GAN")
    print("   4. Validar estrategias contra escenarios adversos")
    print("="*80)
    
    try:
        # === FASE 1: OBTENER DATOS HISTÓRICOS ===
        print("\n📊 FASE 1: Obteniendo datos históricos...")
        data_nexus = DataNexus()
        
        # Obtener datos para múltiples activos
        historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 300)
        
        if historical_data is None or len(historical_data) < 100:
            print("❌ Datos insuficientes para entrenamiento GAN")
            return
        
        print(f"✅ Datos obtenidos: {len(historical_data)} registros históricos")
        
        # === FASE 2: ENTRENAR MODELO GAN ===
        print("\n🧠 FASE 2: Entrenando modelo TimeGAN para mercados sintéticos...")
        print("⚡ Versión optimizada para demostración")
        
        # Entrenar GAN con parámetros optimizados para velocidad
        training_results = await train_market_gan(
            historical_data=historical_data,
            sequence_length=100,  # Secuencias medianas
            epochs=25  # Entrenamiento balanceado
        )
        
        print("✅ Entrenamiento GAN completado")
        print(f"   📈 Generator Loss: {training_results['final_losses']['generator']:.4f}")
        print(f"   📈 Discriminator Loss: {training_results['final_losses']['discriminator']:.4f}")
        
        # === FASE 3: GENERAR ESCENARIOS SINTÉTICOS ===
        print("\n🎲 FASE 3: Generando escenarios sintéticos adversos...")
        
        synthetic_scenarios = generate_scenarios(
            num_scenarios=50,  # Cantidad moderada para demo
            scenario_length=100
        )
        
        print(f"✅ Generados {len(synthetic_scenarios)} escenarios sintéticos")
        
        # Mostrar estadísticas de los escenarios
        if synthetic_scenarios:
            sample_scenario = synthetic_scenarios[0]
            print(f"   📊 Ejemplo de escenario: {len(sample_scenario)} velas")
            print(f"   💰 Rango de precios: ${sample_scenario['close'].min():.2f} - ${sample_scenario['close'].max():.2f}")
        
        # === FASE 4: PROBAR ESTRATEGIA SIMPLE ===
        print("\n🧪 FASE 4: Probando estrategia contra escenarios sintéticos...")
        
        test_strategy = {
            'indicator': 'RSI',
            'entry_level': 30,
            'exit_level': 70,
            'rsi_period': 14,
            'timeframe': '1h'
        }
        
        print(f"🎯 Estrategia de prueba: {test_strategy}")
        
        # Ejecutar stress test
        stress_results = stress_test_strategy(test_strategy, num_scenarios=len(synthetic_scenarios))
        
        print("\n📊 RESULTADOS DEL STRESS TEST:")
        print(f"   ✅ Escenarios exitosos: {stress_results['successful_tests']}/{stress_results['total_scenarios']}")
        print(f"   📈 Tasa de éxito: {stress_results['success_rate']:.1%}")
        
        if 'robustness_score' in stress_results:
            print(f"   🛡️ Score de robustez: {stress_results['robustness_score']:.2f}/100")
            
        if 'win_rate_stats' in stress_results:
            print(f"   🎯 Win rate promedio: {stress_results['win_rate_stats']['mean']:.1f}%")
            print(f"   📊 Desviación estándar: {stress_results['win_rate_stats']['std']:.1f}%")
            print(f"   ⬇️ Win rate mínimo: {stress_results['win_rate_stats']['min']:.1f}%")
            print(f"   ⬆️ Win rate máximo: {stress_results['win_rate_stats']['max']:.1f}%")
        
        # === FASE 5: EVOLUCIÓN CON STRESS TESTING ===
        print("\n🧬 FASE 5: Evolucionando estrategias con stress testing GAN...")
        print("🎯 El motor evolutivo ahora evaluará estrategias contra escenarios sintéticos")
        
        # Evolucionar estrategia con sistema integrado
        best_evolved_strategy = await evolve_trading_strategy(
            pair='BTCUSDT',
            timeframe='1h',
            generations=5,  # Reducido para demo
            population_size=20  # Población pequeña para demo
        )
        
        if best_evolved_strategy:
            print("\n🏆 ESTRATEGIA EVOLUCIONADA CON STRESS TESTING:")
            print(f"   🧬 Estrategia: {best_evolved_strategy}")
            
            # Probar estrategia evolucionada contra escenarios sintéticos
            print("\n🔬 Validando estrategia evolucionada contra GAN...")
            evolved_stress_results = stress_test_strategy(best_evolved_strategy, num_scenarios=30)
            
            if 'robustness_score' in evolved_stress_results:
                print(f"   🛡️ Robustez de estrategia evolucionada: {evolved_stress_results['robustness_score']:.2f}/100")
                
                if evolved_stress_results['robustness_score'] > stress_results.get('robustness_score', 0):
                    print("   🚀 ¡La evolución mejoró la robustez de la estrategia!")
                else:
                    print("   📊 La estrategia original era más robusta")
        
        print("\n🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("✅ Sistema GAN + Motor Evolutivo integrado y funcionando")
        print("🧠 Las estrategias ahora son evaluadas contra futuros sintéticos adversos")
        print("🛡️ Garantiza mayor robustez en condiciones de mercado impredecibles")
        
    except Exception as e:
        print(f"❌ Error en demostración: {str(e)}")
        import traceback
        traceback.print_exc()

async def demo_quick_stress_test():
    """
    Demo rápido enfocado únicamente en stress testing.
    """
    print("⚡ DEMO RÁPIDO: STRESS TESTING CON ESCENARIOS SINTÉTICOS")
    print("="*60)
    
    try:
        # Obtener datos mínimos
        data_nexus = DataNexus()
        historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 150)
        
        if historical_data is None:
            print("❌ No se pudieron obtener datos")
            return
        
        # Entrenar GAN ultra-rápido
        print("🧠 Entrenamiento GAN express...")
        await train_market_gan(
            historical_data=historical_data,
            sequence_length=50,
            epochs=15
        )
        
        # Generar escenarios mínimos
        print("🎲 Generando escenarios...")
        synthetic_scenarios = generate_scenarios(num_scenarios=20, scenario_length=50)
        
        # Probar estrategia
        strategy = {'indicator': 'RSI', 'entry_level': 25, 'exit_level': 75, 'rsi_period': 14}
        stress_results = stress_test_strategy(strategy, num_scenarios=len(synthetic_scenarios))
        
        print(f"📊 RESULTADO: Robustez {stress_results.get('robustness_score', 0):.1f}/100")
        print(f"✅ Demo rápido completado")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(demo_quick_stress_test())
    else:
        asyncio.run(demo_complete_gan_integration())
