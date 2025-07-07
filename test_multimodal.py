
#!/usr/bin/env python3
"""
Test script for Multimodal Analyzer
Prueba las capacidades de análisis de video y audio del sistema.
"""

import asyncio
from multimodal_analyzer import MultimodalAnalyzer


async def test_multimodal_integration():
    """
    Prueba la integración del análisis multimodal.
    """
    print("🎥 INICIANDO PRUEBA DEL ANÁLISIS MULTIMODAL")
    print("="*60)
    
    # Inicializar analizador
    analyzer = MultimodalAnalyzer()
    
    # Casos de prueba con diferentes contextos
    test_cases = [
        {
            'keywords': 'Bitcoin news today analysis',
            'context': 'crypto_news',
            'description': 'Análisis de noticias de Bitcoin'
        },
        {
            'keywords': 'Fed Powell speech latest',
            'context': 'fed_speech',
            'description': 'Discurso de política monetaria de la Fed'
        },
        {
            'keywords': 'Apple earnings call Q4',
            'context': 'earnings_call',
            'description': 'Llamada de resultados corporativos'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 CASO DE PRUEBA {i}: {test_case['description']}")
        print("-" * 40)
        
        try:
            # Realizar análisis multimodal
            result = await analyzer.analyze_financial_video_by_keywords(
                test_case['keywords'], 
                max_results=1
            )
            
            print(f"✅ Resultado: {result}")
            
        except Exception as e:
            print(f"❌ Error en caso de prueba {i}: {e}")
        
        # Pausa entre pruebas
        await asyncio.sleep(2)
    
    print("="*60)


async def test_direct_video_analysis():
    """
    Prueba análisis directo de una URL de video específica.
    """
    print("\n🎬 PRUEBA DE ANÁLISIS DIRECTO DE VIDEO")
    print("-" * 40)
    
    analyzer = MultimodalAnalyzer()
    
    # URL de ejemplo (usar una URL pública de noticias financieras)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # URL de ejemplo
    
    try:
        result = await analyzer.analyze_video_url(test_url, 'financial_news')
        print(f"✅ Análisis directo: {result}")
    except Exception as e:
        print(f"❌ Error en análisis directo: {e}")


async def test_prompt_variations():
    """
    Prueba diferentes variaciones de prompts multimodales.
    """
    print("\n🎯 PRUEBA DE VARIACIONES DE PROMPTS")
    print("-" * 40)
    
    analyzer = MultimodalAnalyzer()
    
    # Probar diferentes contextos
    contexts = ['financial_news', 'fed_speech', 'earnings_call', 'crypto_news']
    
    for context in contexts:
        print(f"\n📝 Contexto: {context}")
        prompt = analyzer._create_multimodal_prompt(context)
        print(f"Prompt generado: {len(prompt)} caracteres")
        print(f"Extracto: {prompt[:150]}...")


if __name__ == "__main__":
    print("🚀 Iniciando pruebas del sistema multimodal...")
    
    # Ejecutar pruebas
    asyncio.run(test_multimodal_integration())
    asyncio.run(test_direct_video_analysis())
    asyncio.run(test_prompt_variations())
    
    print("\n✅ Pruebas del Multimodal Analyzer completadas.")
