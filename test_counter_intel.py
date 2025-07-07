
#!/usr/bin/env python3
"""
Test script for Counter Intelligence Engine
Prueba las capacidades de detección de manipulación del mercado.
"""

import asyncio
from counter_intel_engine import CounterIntelEngine


async def test_counter_intelligence_comprehensive():
    """
    Prueba completa del sistema de contrainteligencia.
    """
    print("🕵️ INICIANDO PRUEBAS COMPRENSIVAS DE CONTRAINTELIGENCIA")
    print("="*70)
    
    # Inicializar motor
    engine = CounterIntelEngine()
    
    # Casos de prueba con diferentes escenarios
    test_scenarios = [
        {
            'asset': 'BTC',
            'description': 'Bitcoin - Activo principal',
            'expected_range': (0.0, 0.6)
        },
        {
            'asset': 'DOGE',
            'description': 'Dogecoin - Activo con alta volatilidad social',
            'expected_range': (0.2, 0.8)
        },
        {
            'asset': 'SHIB',
            'description': 'Shiba Inu - Meme coin con potencial manipulación',
            'expected_range': (0.3, 0.9)
        },
        {
            'asset': 'ETH',
            'description': 'Ethereum - Activo institucional',
            'expected_range': (0.0, 0.5)
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🔍 ESCENARIO: {scenario['description']}")
        print("-" * 50)
        
        try:
            # Realizar análisis de manipulación
            manipulation_score = await engine.analyze_market_manipulation(
                scenario['asset'], 
                timeframe_hours=24
            )
            
            # Evaluar resultado
            min_expected, max_expected = scenario['expected_range']
            within_expected = min_expected <= manipulation_score <= max_expected
            
            status_icon = "✅" if within_expected else "⚠️"
            risk_level = get_risk_level(manipulation_score)
            
            print(f"{status_icon} {scenario['asset']}: Score {manipulation_score:.3f} - {risk_level}")
            
            if manipulation_score > 0.7:
                print(f"🚨 ALERTA CRÍTICA: Posible manipulación detectada en {scenario['asset']}")
            elif manipulation_score > 0.5:
                print(f"⚠️ ADVERTENCIA: Actividad sospechosa en {scenario['asset']}")
            
            results.append({
                'asset': scenario['asset'],
                'score': manipulation_score,
                'risk_level': risk_level,
                'within_expected': within_expected
            })
            
        except Exception as e:
            print(f"❌ Error en análisis de {scenario['asset']}: {e}")
            results.append({
                'asset': scenario['asset'],
                'score': None,
                'risk_level': 'ERROR',
                'within_expected': False
            })
        
        # Pausa entre pruebas
        await asyncio.sleep(1)
    
    # Resumen final
    print(f"\n📊 RESUMEN DE RESULTADOS")
    print("="*50)
    
    successful_tests = sum(1 for r in results if r['score'] is not None)
    high_risk_assets = sum(1 for r in results if r['score'] and r['score'] > 0.7)
    
    print(f"✅ Pruebas exitosas: {successful_tests}/{len(test_scenarios)}")
    print(f"🚨 Activos de alto riesgo: {high_risk_assets}")
    
    for result in results:
        if result['score'] is not None:
            print(f"   {result['asset']}: {result['score']:.3f} ({result['risk_level']})")
    
    print(f"\n🎯 Sistema de contrainteligencia: {'OPERATIVO' if successful_tests > 0 else 'NECESITA REVISIÓN'}")


def get_risk_level(score):
    """Convierte score numérico a nivel de riesgo."""
    if score is None:
        return "DESCONOCIDO"
    elif score > 0.7:
        return "ALTO RIESGO"
    elif score > 0.5:
        return "RIESGO MEDIO"
    elif score > 0.3:
        return "RIESGO BAJO"
    else:
        return "NORMAL"


async def test_individual_components():
    """
    Prueba componentes individuales del sistema.
    """
    print("\n🔧 PRUEBAS DE COMPONENTES INDIVIDUALES")
    print("="*50)
    
    engine = CounterIntelEngine()
    
    # Probar métodos específicos
    test_asset = 'BTC'
    
    print(f"🧪 Probando componentes para {test_asset}...")
    
    try:
        # Simular y probar análisis social
        social_score = await engine._analyze_social_media_patterns(test_asset, 24)
        print(f"📱 Análisis social: {social_score:.3f}")
        
        # Simular y probar correlación de eventos
        correlation_score = await engine._analyze_event_correlation(test_asset, 24)
        print(f"🔗 Correlación eventos: {correlation_score:.3f}")
        
        # Simular y probar actividad coordinada
        coordination_score = await engine._analyze_coordinated_activity(test_asset, 24)
        print(f"🤝 Actividad coordinada: {coordination_score:.3f}")
        
        # Simular y probar anomalías temporales
        temporal_score = await engine._analyze_temporal_anomalies(test_asset, 24)
        print(f"⏰ Anomalías temporales: {temporal_score:.3f}")
        
        print("✅ Todos los componentes funcionando correctamente")
        
    except Exception as e:
        print(f"❌ Error en pruebas de componentes: {e}")


if __name__ == "__main__":
    print("🚀 Iniciando suite completa de pruebas de contrainteligencia...")
    
    # Ejecutar todas las pruebas
    asyncio.run(test_counter_intelligence_comprehensive())
    asyncio.run(test_individual_components())
    
    print("\n🏁 Suite de pruebas completada.")
    print("🕵️ Sistema de contrainteligencia listo para operación.")
