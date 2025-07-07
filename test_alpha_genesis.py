
#!/usr/bin/env python3
"""
Test script para el motor Alpha Genesis.
Demuestra el descubrimiento autónomo de indicadores técnicos.
"""

import asyncio
from alpha_genesis_engine import AlphaGenesisEngine, discover_trading_indicators, demo_alpha_genesis
from datanexus_connector import DataNexus

async def test_alpha_genesis_basic():
    """
    Prueba básica del motor Alpha Genesis.
    """
    print("🧬 TEST: Alpha Genesis - Descubrimiento Básico")
    print("="*60)
    
    try:
        # Inicializar motor
        alpha_genesis = AlphaGenesisEngine()
        
        # Test de generación de datos sintéticos
        print("📊 Generando datos sintéticos de prueba...")
        test_data = alpha_genesis._generate_synthetic_test_data(100)
        
        if test_data is not None:
            print(f"✅ Datos sintéticos generados: {len(test_data)} velas")
            print(f"📈 Rango de precios: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        else:
            print("❌ Error generando datos sintéticos")
            return False
        
        # Test de descubrimiento con parámetros reducidos
        print("\n🔬 Iniciando descubrimiento de indicadores (versión rápida)...")
        indicators = await alpha_genesis.discover_indicators(
            generations=5,     # Muy reducido para test rápido
            population_size=10, # Muy reducido para test rápido
            pair='BTCUSDT',
            timeframe='1h'
        )
        
        if indicators:
            print(f"✅ Descubrimiento exitoso: {len(indicators)} indicadores encontrados")
            
            for i, indicator in enumerate(indicators[:3], 1):
                print(f"   {i}. Fitness: {indicator['fitness']:.4f}")
                print(f"      ID: {indicator['id']}")
                print(f"      Expresión: {indicator['expression'][:60]}...")
            
            return True
        else:
            print("⚠️ No se encontraron indicadores en esta ejecución")
            return False
            
    except Exception as e:
        print(f"❌ Error en test básico: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_alpha_genesis_integration():
    """
    Prueba la integración de Alpha Genesis con otros componentes.
    """
    print("\n🔗 TEST: Integración con EvolutionaryEngine")
    print("="*60)
    
    try:
        from evolutionary_engine import EvolutionaryEngine
        from datanexus_connector import DataNexus
        
        # Inicializar componentes
        data_nexus = DataNexus()
        evolution_engine = EvolutionaryEngine(data_nexus)
        
        # Test de obtención de indicadores Alpha Genesis
        print("🧬 Probando integración Alpha Genesis...")
        alpha_indicators = evolution_engine._get_alpha_genesis_indicators()
        
        if alpha_indicators:
            print(f"✅ Integración exitosa: {len(alpha_indicators)} indicadores disponibles")
            
            # Test de configuración de indicador
            test_indicator = alpha_indicators[0]
            config = evolution_engine._get_alpha_genesis_config(test_indicator)
            
            print(f"📊 Configuración de prueba:")
            print(f"   Alpha ID: {config.get('alpha_id', 'N/A')}")
            print(f"   Fitness: {config.get('alpha_fitness', 0):.4f}")
            print(f"   Expresión: {config.get('alpha_expression', 'N/A')[:50]}...")
            
            return True
        else:
            print("⚠️ No hay indicadores Alpha Genesis disponibles para integración")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de integración: {str(e)}")
        return False

async def test_alpha_genesis_with_real_data():
    """
    Prueba Alpha Genesis con datos reales (si están disponibles).
    """
    print("\n📊 TEST: Alpha Genesis con Datos Reales")
    print("="*60)
    
    try:
        from datanexus_connector import DataNexus
        
        # Inicializar conector de datos
        data_nexus = DataNexus()
        
        # Intentar obtener datos reales
        print("📥 Obteniendo datos reales de BTC-USDT...")
        real_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 200)
        
        if real_data is not None and len(real_data) > 50:
            print(f"✅ Datos reales obtenidos: {len(real_data)} velas")
            
            # Descubrir indicadores con datos reales
            alpha_genesis = AlphaGenesisEngine(data_nexus)
            
            print("🔬 Descubriendo indicadores con datos reales...")
            indicators = await alpha_genesis.discover_indicators(
                generations=8,     # Moderado para test con datos reales
                population_size=15,
                pair='BTCUSDT',
                timeframe='1h'
            )
            
            if indicators:
                print(f"✅ Descubrimiento con datos reales exitoso: {len(indicators)} indicadores")
                
                # Mostrar el mejor indicador
                best = max(indicators, key=lambda x: x['fitness'])
                print(f"\n🏆 MEJOR INDICADOR DESCUBIERTO:")
                print(f"   Fitness: {best['fitness']:.4f}")
                print(f"   ID: {best['id']}")
                print(f"   Expresión: {best['expression'][:80]}...")
                
                return True
            else:
                print("⚠️ No se encontraron indicadores con datos reales")
                return False
        else:
            print("⚠️ No se pudieron obtener datos reales, saltando test")
            return True  # No es un fallo, solo falta de datos
            
    except Exception as e:
        print(f"❌ Error en test con datos reales: {str(e)}")
        return False

async def test_alpha_genesis_database():
    """
    Prueba las funcionalidades de base de datos de Alpha Genesis.
    """
    print("\n💾 TEST: Base de Datos Alpha Genesis")
    print("="*60)
    
    try:
        alpha_genesis = AlphaGenesisEngine()
        
        # Test de guardado manual
        test_indicator = {
            'id': 'test_alpha_genesis_001',
            'expression': '(high + low) / 2 - close * 1.5',
            'fitness': 0.75,
            'generation': 1,
            'discovery_date': '2024-01-01T12:00:00'
        }
        
        print("💾 Guardando indicador de prueba...")
        alpha_genesis._save_indicator(test_indicator)
        
        # Test de recuperación
        print("📋 Recuperando mejores indicadores...")
        best_indicators = alpha_genesis.get_best_indicators(5)
        
        if best_indicators:
            print(f"✅ Base de datos funcionando: {len(best_indicators)} indicadores encontrados")
            
            for i, indicator in enumerate(best_indicators[:3], 1):
                print(f"   {i}. Fitness: {indicator['fitness']:.4f}")
                print(f"      Fecha: {indicator['discovery_date'][:10]}")
                print(f"      Expresión: {indicator['expression'][:50]}...")
            
            # Test de exportación
            test_export = alpha_genesis.export_indicator_for_evolution(best_indicators[0]['id'])
            if test_export:
                print("✅ Exportación para evolución exitosa")
                print(f"   Fitness Score: {test_export['fitness_score']:.4f}")
            else:
                print("⚠️ Error en exportación para evolución")
            
            return True
        else:
            print("⚠️ No se encontraron indicadores en la base de datos")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de base de datos: {str(e)}")
        return False

async def main():
    """
    Función principal para ejecutar todos los tests de Alpha Genesis.
    """
    print("🧬 INICIANDO TESTS DEL MOTOR ALPHA GENESIS")
    print("="*70)
    print("🎯 Pruebas de descubrimiento autónomo de indicadores técnicos")
    print("="*70)
    
    tests_results = []
    
    # Test básico
    result1 = await test_alpha_genesis_basic()
    tests_results.append(("Test Básico", result1))
    
    # Test de base de datos
    result2 = await test_alpha_genesis_database()
    tests_results.append(("Test Base de Datos", result2))
    
    # Test de integración
    result3 = await test_alpha_genesis_integration()
    tests_results.append(("Test Integración", result3))
    
    # Test con datos reales
    result4 = await test_alpha_genesis_with_real_data()
    tests_results.append(("Test Datos Reales", result4))
    
    # Demo completo
    print("\n🎭 EJECUTANDO DEMO COMPLETO...")
    try:
        await demo_alpha_genesis()
        tests_results.append(("Demo Completo", True))
    except Exception as e:
        print(f"❌ Error en demo: {str(e)}")
        tests_results.append(("Demo Completo", False))
    
    # Resumen final
    print("\n📊 RESUMEN DE TESTS ALPHA GENESIS")
    print("="*70)
    
    passed = 0
    total = len(tests_results)
    
    for test_name, result in tests_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("="*70)
    print(f"🎯 RESULTADO FINAL: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 ¡Todos los tests de Alpha Genesis fueron exitosos!")
        print("🧬 El motor de descubrimiento autónomo está listo para usar")
    else:
        print("⚠️ Algunos tests fallaron. Revisar la implementación.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
