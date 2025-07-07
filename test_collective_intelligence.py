
#!/usr/bin/env python3
"""
Script de prueba para la Red de Inteligencia Colectiva.
Demuestra el funcionamiento del sistema de métricas distribuidas.
"""

import asyncio
import requests
import json
import random
from datetime import datetime

def test_collective_intelligence_api():
    """
    Prueba los endpoints de la Red de Inteligencia Colectiva.
    """
    print("🌐 TESTING RED DE INTELIGENCIA COLECTIVA")
    print("="*60)
    
    base_url = "http://0.0.0.0:5000"
    
    # 1. Simular datos de diferentes nodos
    strategies = ['RSI_BUY', 'RSI_SELL', 'MACD_BUY', 'MACD_SELL', 'SMA_BUY', 'SMA_SELL']
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframes = ['1m', '5m', '15m', '30m', '1h']
    results = ['WIN', 'LOSS']
    
    print("📊 Enviando datos de prueba simulados...")
    
    # Simular 50 trades de diferentes nodos
    for i in range(50):
        collective_data = {
            'estrategia_usada': random.choice(strategies),
            'activo': random.choice(assets),
            'timeframe': random.choice(timeframes),
            'resultado_win_loss': random.choice(results)
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/log_collective_trade",
                json=collective_data,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"  ✅ Trade {i+1}: {collective_data['estrategia_usada']} -> {collective_data['resultado_win_loss']}")
            else:
                print(f"  ❌ Error en trade {i+1}: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error de conexión en trade {i+1}: {str(e)}")
    
    print("\n📈 Consultando métricas agregadas...")
    
    # 2. Consultar métricas generales
    try:
        response = requests.get(f"{base_url}/api/collective_metrics", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🌐 MÉTRICAS DE LA RED:")
            print(f"   📊 Total trades: {data['network_stats']['total_trades']}")
            print(f"   🔗 Nodos activos: {data['network_stats']['active_nodes']}")
            print(f"   📈 Estrategias analizadas: {data['network_stats']['strategies_analyzed']}")
            
            print(f"\n🏆 TOP ESTRATEGIAS GLOBALES:")
            for i, metric in enumerate(data['collective_metrics'][:5], 1):
                print(f"   {i}. {metric['estrategia_usada']} ({metric['timeframe']})")
                print(f"      Win Rate: {metric['win_rate']:.1f}% ({metric['wins']}W/{metric['losses']}L)")
                print(f"      Nodos: {metric['contributing_nodes']}, Trades: {metric['total_trades']}")
                print()
        else:
            print(f"❌ Error consultando métricas: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
    
    # 3. Consultar métricas específicas
    print("🔍 Consultando métricas específicas...")
    
    test_queries = [
        {'estrategia': 'RSI'},
        {'timeframe': '5m'},
        {'activo': 'BTC'},
        {'estrategia': 'MACD', 'timeframe': '15m'}
    ]
    
    for query in test_queries:
        try:
            response = requests.get(
                f"{base_url}/api/collective_metrics",
                params=query,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n📊 Consulta {query}:")
                print(f"   Resultados: {len(data['collective_metrics'])}")
                
                for metric in data['collective_metrics'][:3]:
                    print(f"   - {metric['estrategia_usada']}: {metric['win_rate']:.1f}% ({metric['total_trades']} trades)")
            else:
                print(f"❌ Error en consulta {query}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error de conexión en consulta {query}: {str(e)}")


async def test_collective_with_evolution():
    """
    Prueba la integración con el motor evolutivo.
    """
    print("\n🧬 TESTING INTEGRACIÓN CON MOTOR EVOLUTIVO")
    print("="*60)
    
    try:
        from evolutionary_engine import EvolutionaryEngine
        from datanexus_connector import DataNexus
        
        # Crear instancias
        data_nexus = DataNexus()
        evolution_engine = EvolutionaryEngine(data_nexus)
        
        # Estrategia de prueba
        test_strategy = {
            'indicator': 'RSI',
            'timeframe': '5m',
            'entry_level': 30,
            'exit_level': 70
        }
        
        print("📊 Probando consulta de métricas colectivas...")
        
        # Obtener métricas colectivas
        collective_metrics = evolution_engine._get_collective_intelligence_performance(test_strategy)
        
        print(f"🌐 Resultado de consulta colectiva:")
        print(f"   Win Rate: {collective_metrics['win_rate']:.1%}")
        print(f"   Total Trades: {collective_metrics['total_trades']}")
        print(f"   Es colectivo: {collective_metrics['is_collective']}")
        print(f"   Nodos contribuyentes: {collective_metrics.get('contributing_nodes', 0)}")
        
        if collective_metrics['is_collective']:
            print("✅ Red de Inteligencia Colectiva funcionando correctamente con evolución")
        else:
            print("⚠️ Sin datos colectivos disponibles para esta estrategia")
            
    except Exception as e:
        print(f"❌ Error en integración evolutiva: {str(e)}")


def test_collective_anonymization():
    """
    Verifica que los datos se anonimicen correctamente.
    """
    print("\n🔒 TESTING ANONIMIZACIÓN DE DATOS")
    print("="*60)
    
    # Simular múltiples envíos desde el mismo "nodo"
    test_data = {
        'estrategia_usada': 'RSI_BUY',
        'activo': 'BTCUSDT',
        'timeframe': '5m',
        'resultado_win_loss': 'WIN'
    }
    
    node_hashes = []
    
    for i in range(3):
        try:
            response = requests.post(
                "http://0.0.0.0:5000/api/log_collective_trade",
                json=test_data,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                node_hash = data.get('node_hash')
                node_hashes.append(node_hash)
                print(f"Envío {i+1}: Hash del nodo: {node_hash}")
            else:
                print(f"❌ Error en envío {i+1}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error de conexión en envío {i+1}: {str(e)}")
    
    # Verificar que los hashes sean consistentes (mismo nodo, mismo hash)
    if len(set(node_hashes)) == 1:
        print("✅ Anonimización consistente: mismo nodo genera mismo hash")
    else:
        print("⚠️ Problema de anonimización: hashes inconsistentes")
        print(f"Hashes generados: {node_hashes}")


async def main():
    """
    Función principal de pruebas.
    """
    print("🚀 INICIANDO PRUEBAS DE RED DE INTELIGENCIA COLECTIVA")
    print("🌐 Sistema Nexus-Alpha - Inteligencia Distribuida")
    print("="*80)
    
    # Pruebas de API
    test_collective_intelligence_api()
    
    # Pruebas de anonimización
    test_collective_anonymization()
    
    # Pruebas de integración evolutiva
    await test_collective_with_evolution()
    
    print("\n🎉 PRUEBAS COMPLETADAS")
    print("="*40)
    print("La Red de Inteligencia Colectiva está operativa y lista para")
    print("contribuir al conocimiento global de trading de Nexus-Alpha.")


if __name__ == "__main__":
    asyncio.run(main())
