
#!/usr/bin/env python3
"""
Pruebas avanzadas para el sistema de análisis multimodal
Valida diferentes escenarios de contenido económico y financiero.
"""

import asyncio
from multimodal_analyzer import MultimodalAnalyzer
from deep_analysis_engine import DeepAnalysisEngine
from datanexus_connector import DataNexus


async def test_economic_announcement_analysis():
    """
    Prueba análisis de anuncios económicos específicos.
    """
    print("🏛️ PRUEBA DE ANÁLISIS DE ANUNCIOS ECONÓMICOS")
    print("="*60)
    
    analyzer = MultimodalAnalyzer()
    
    # Casos específicos de anuncios económicos
    economic_scenarios = [
        {
            'keywords': 'Federal Reserve interest rate decision Powell',
            'context': 'fed_speech',
            'description': 'Decisión de tasas de interés de la Fed'
        },
        {
            'keywords': 'ECB Christine Lagarde monetary policy',
            'context': 'fed_speech',
            'description': 'Política monetaria del BCE'
        },
        {
            'keywords': 'Tesla earnings call Elon Musk Q4 results',
            'context': 'earnings_call',
            'description': 'Resultados trimestrales de Tesla'
        },
        {
            'keywords': 'Bitcoin ETF approval SEC announcement',
            'context': 'crypto_news',
            'description': 'Aprobación de ETF de Bitcoin'
        }
    ]
    
    results = []
    
    for scenario in economic_scenarios:
        print(f"\n📺 Analizando: {scenario['description']}")
        print(f"🔍 Búsqueda: {scenario['keywords']}")
        
        try:
            result = await analyzer.analyze_financial_video_by_keywords(
                scenario['keywords'],
                max_results=1
            )
            
            # Extraer indicadores clave
            confidence_indicators = extract_confidence_indicators(result)
            
            results.append({
                'scenario': scenario['description'],
                'result': result,
                'confidence_indicators': confidence_indicators
            })
            
            print(f"✅ Resultado: {result[:100]}...")
            print(f"🎯 Indicadores: {confidence_indicators}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(3)  # Pausa para evitar rate limiting
    
    return results


def extract_confidence_indicators(multimodal_result: str) -> dict:
    """
    Extrae indicadores de confianza del resultado multimodal.
    """
    indicators = {
        'confianza_detectada': False,
        'evasion_detectada': False,
        'coherencia_multimodal': False,
        'urgencia_detectada': False
    }
    
    result_lower = multimodal_result.lower()
    
    # Detectar confianza
    confidence_words = ['confianza', 'seguridad', 'certeza', 'firme', 'decidido']
    indicators['confianza_detectada'] = any(word in result_lower for word in confidence_words)
    
    # Detectar evasión
    evasion_words = ['evasión', 'duda', 'nerviosismo', 'titubeo', 'incertidumbre']
    indicators['evasion_detectada'] = any(word in result_lower for word in evasion_words)
    
    # Detectar coherencia
    coherence_words = ['coherencia', 'alineación', 'consistente', 'coincide']
    indicators['coherencia_multimodal'] = any(word in result_lower for word in coherence_words)
    
    # Detectar urgencia
    urgency_words = ['urgencia', 'rápido', 'inmediato', 'crítico']
    indicators['urgencia_detectada'] = any(word in result_lower for word in urgency_words)
    
    return indicators


async def test_full_integration():
    """
    Prueba la integración completa del análisis multimodal en el sistema de decisiones.
    """
    print("\n🔗 PRUEBA DE INTEGRACIÓN COMPLETA")
    print("="*60)
    
    # Simular una oportunidad de trading
    test_opportunity = {
        'pair': 'BTC-USDT',
        'type': 'crypto',
        'timeframe': '1h',
        'reason': 'RSI en sobreventa (25.3) + Volume Spike detectado'
    }
    
    print(f"📊 Oportunidad de prueba: {test_opportunity}")
    
    try:
        # Inicializar componentes (sin socketio para pruebas)
        data_connector = DataNexus()
        
        # Crear un mock simple de socketio para pruebas
        class MockSocketIO:
            def emit(self, event, data): pass
        
        mock_socketio = MockSocketIO()
        
        # Solo probar el análisis multimodal específico
        analyzer = MultimodalAnalyzer(mock_socketio)
        
        # Crear keywords basadas en la oportunidad
        base_asset = test_opportunity['pair'].split('-')[0]
        search_keywords = f"{base_asset} cryptocurrency news today analysis"
        
        print(f"🔍 Analizando contenido multimodal para: {search_keywords}")
        
        multimodal_result = await analyzer.analyze_financial_video_by_keywords(
            search_keywords, max_results=1
        )
        
        print(f"✅ Análisis multimodal completado:")
        print(f"   {multimodal_result}")
        
        # Evaluar el impacto en la decisión
        impact_assessment = assess_decision_impact(multimodal_result)
        print(f"📈 Impacto en decisión: {impact_assessment}")
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")


def assess_decision_impact(multimodal_result: str) -> str:
    """
    Evalúa el impacto del análisis multimodal en la decisión de trading.
    """
    result_lower = multimodal_result.lower()
    
    # Factores positivos
    positive_factors = ['confianza', 'positivo', 'alcista', 'optimista', 'coherencia']
    positive_count = sum(1 for factor in positive_factors if factor in result_lower)
    
    # Factores negativos
    negative_factors = ['evasión', 'duda', 'negativo', 'bajista', 'nerviosismo']
    negative_count = sum(1 for factor in negative_factors if factor in result_lower)
    
    # Factores de alerta
    alert_factors = ['incongruencia', 'inconsistente', 'contradictorio']
    alert_count = sum(1 for factor in alert_factors if factor in result_lower)
    
    if alert_count > 0:
        return "ALERTA CRÍTICA - Incongruencias detectadas"
    elif positive_count > negative_count:
        return f"IMPACTO POSITIVO - {positive_count} factores positivos vs {negative_count} negativos"
    elif negative_count > positive_count:
        return f"IMPACTO NEGATIVO - {negative_count} factores negativos vs {positive_count} positivos"
    else:
        return "IMPACTO NEUTRAL - Factores balanceados"


async def test_prompt_effectiveness():
    """
    Prueba la efectividad de diferentes prompts multimodales.
    """
    print("\n🎯 PRUEBA DE EFECTIVIDAD DE PROMPTS")
    print("="*60)
    
    analyzer = MultimodalAnalyzer()
    
    contexts = ['financial_news', 'fed_speech', 'earnings_call', 'crypto_news']
    
    for context in contexts:
        print(f"\n📝 Contexto: {context.upper()}")
        
        prompt = analyzer._create_multimodal_prompt(context)
        
        # Análizar características del prompt
        prompt_analysis = {
            'length': len(prompt),
            'financial_keywords': count_financial_keywords(prompt),
            'multimodal_elements': count_multimodal_elements(prompt),
            'instruction_clarity': assess_instruction_clarity(prompt)
        }
        
        print(f"   📏 Longitud: {prompt_analysis['length']} caracteres")
        print(f"   💰 Keywords financieras: {prompt_analysis['financial_keywords']}")
        print(f"   🎥 Elementos multimodales: {prompt_analysis['multimodal_elements']}")
        print(f"   📋 Claridad instruccional: {prompt_analysis['instruction_clarity']}")


def count_financial_keywords(prompt: str) -> int:
    """Cuenta keywords financieras en el prompt."""
    financial_terms = [
        'financiero', 'económico', 'trading', 'mercado', 'inversión',
        'proyecciones', 'resultados', 'earnings', 'fed', 'central bank'
    ]
    return sum(1 for term in financial_terms if term.lower() in prompt.lower())


def count_multimodal_elements(prompt: str) -> int:
    """Cuenta elementos multimodales mencionados en el prompt."""
    multimodal_terms = [
        'lenguaje corporal', 'tono de voz', 'gestos', 'postura', 
        'contacto visual', 'expresiones', 'coherencia', 'verbal'
    ]
    return sum(1 for term in multimodal_terms if term.lower() in prompt.lower())


def assess_instruction_clarity(prompt: str) -> str:
    """Evalúa la claridad de las instrucciones."""
    instruction_indicators = ['analiza', 'considera', 'evalúa', 'proporciona', 'responde']
    clarity_score = sum(1 for indicator in instruction_indicators if indicator.lower() in prompt.lower())
    
    if clarity_score >= 4:
        return "ALTA"
    elif clarity_score >= 2:
        return "MEDIA"
    else:
        return "BAJA"


if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS AVANZADAS DEL SISTEMA MULTIMODAL")
    print("="*80)
    
    # Ejecutar todas las pruebas
    asyncio.run(test_economic_announcement_analysis())
    asyncio.run(test_full_integration())
    asyncio.run(test_prompt_effectiveness())
    
    print("\n✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("="*80)
