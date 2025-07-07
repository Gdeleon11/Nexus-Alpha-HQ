
#!/usr/bin/env python3
"""
Test para el Motor de IA Explicable (XAI)
Verifica que las explicaciones SHAP se generen correctamente.
"""

import asyncio
from explanation_engine import generate_explanation, get_explanation, test_explanation_engine


async def test_xai_integration():
    """
    Prueba la integración completa del motor XAI.
    """
    print("🔍 PRUEBA COMPLETA DEL MOTOR XAI")
    print("="*60)
    
    # Datos de prueba realistas
    test_scenarios = [
        {
            'name': 'Señal Alcista Fuerte',
            'inputs': {
                'trigger': 'RSI en sobreventa (22.5)',
                'cognitive': 'Análisis Cognitivo (Gemini): Muy positivo',
                'predictive': {'prediction': 'Alcista', 'confidence': 0.85},
                'macro_context': 'Sentimiento: Positivo - Políticas expansivas favorable',
                'social_sentiment': 'Pulso Social: Eufórico',
                'causal': 'Análisis Causal: Alta causalidad - Score: 0.75',
                'correlation': 'Correlación: Favorable',
                'rlhf': 'Confianza del operador: 0.9',
                'multimodal': 'Análisis Multimodal: Confianza detectada en video'
            },
            'decision': 'COMPRAR'
        },
        {
            'name': 'Señal Bajista Fuerte', 
            'inputs': {
                'trigger': 'RSI en sobrecompra (78.2)',
                'cognitive': 'Análisis Cognitivo (Gemini): Muy negativo',
                'predictive': {'prediction': 'Bajista', 'confidence': 0.80},
                'macro_context': 'Sentimiento: Negativo - Políticas restrictivas',
                'social_sentiment': 'Pulso Social: Pánico',
                'causal': 'Análisis Causal: Causalidad moderada - Score: 0.65',
                'correlation': 'Correlación: Desfavorable',
                'rlhf': 'Confianza del operador: 0.8',
                'multimodal': 'Análisis Multimodal: Nerviosismo detectado'
            },
            'decision': 'VENDER'
        },
        {
            'name': 'Señal Mixta',
            'inputs': {
                'trigger': 'RSI neutral (45.0)',
                'cognitive': 'Análisis Cognitivo (Gemini): Neutral',
                'predictive': {'prediction': 'Plana', 'confidence': 0.55},
                'macro_context': 'Sentimiento: Neutral - Incertidumbre',
                'social_sentiment': 'Pulso Social: Neutral',
                'causal': 'Análisis Causal: Baja causalidad - Score: 0.25',
                'correlation': 'Correlación: Neutral',
                'rlhf': 'Confianza del operador: 0.5',
                'multimodal': 'Análisis Multimodal: Sin datos relevantes'
            },
            'decision': 'NO OPERAR'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 ESCENARIO {i}: {scenario['name']}")
        print("-" * 40)
        
        signal_id = f"test-signal-{i:03d}"
        
        # Generar explicación
        explanation = generate_explanation(
            scenario['inputs'], 
            scenario['decision'], 
            signal_id
        )
        
        print(f"✅ Explicación generada para {signal_id}")
        print(f"🎯 Decisión: {explanation.get('decision', 'N/A')}")
        print(f"📊 Factores: {len(explanation.get('feature_names', []))}")
        
        # Verificar que se pueden recuperar
        retrieved = get_explanation(signal_id)
        if retrieved:
            print(f"✅ Explicación recuperada correctamente")
        else:
            print(f"❌ Error recuperando explicación")
        
        # Mostrar interpretación
        interpretation = explanation.get('interpretation', 'No disponible')
        print(f"📝 Interpretación:\n{interpretation[:200]}...")
        
        # Verificar que los gráficos se generaron
        charts = ['force_plot', 'waterfall_chart', 'feature_importance']
        chart_status = []
        for chart in charts:
            if chart in explanation and explanation[chart]:
                chart_status.append(f"✅ {chart}")
            else:
                chart_status.append(f"❌ {chart}")
        
        print(f"📈 Gráficos: {', '.join(chart_status)}")
    
    print("\n" + "="*60)
    print("🎉 PRUEBA DEL MOTOR XAI COMPLETADA")
    print("="*60)


def test_api_endpoint():
    """
    Prueba el endpoint de API para explicaciones.
    """
    print("\n🌐 PRUEBA DEL ENDPOINT API")
    print("-" * 40)
    
    import requests
    
    try:
        # Generar una explicación de prueba
        test_inputs = {
            'trigger': 'RSI en sobreventa (20.0)',
            'cognitive': 'Positivo',
            'predictive': {'prediction': 'Alcista', 'confidence': 0.7},
            'macro_context': 'Favorable',
            'social_sentiment': 'Optimista',
            'causal': 'Score: 0.6',
            'correlation': 'Favorable',
            'rlhf': 'Confianza: 0.8',
            'multimodal': 'Positivo'
        }
        
        signal_id = "api-test-signal"
        generate_explanation(test_inputs, 'COMPRAR', signal_id)
        
        # Probar el endpoint
        url = f"http://0.0.0.0:5000/api/get_explanation/{signal_id}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Endpoint API funcionando correctamente")
            print(f"📊 Status: {data.get('status', 'N/A')}")
            print(f"🎯 Decisión en respuesta: {data.get('explanation', {}).get('decision', 'N/A')}")
        else:
            print(f"❌ Error en endpoint: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error probando endpoint: {e}")


async def main():
    """
    Función principal de pruebas.
    """
    print("🔍 SISTEMA DE PRUEBAS PARA MOTOR XAI")
    print("="*60)
    
    # Prueba básica del motor
    test_explanation_engine()
    
    # Prueba de integración completa
    await test_xai_integration()
    
    # Prueba del endpoint API
    test_api_endpoint()
    
    print("\n🎊 TODAS LAS PRUEBAS COMPLETADAS")


if __name__ == "__main__":
    asyncio.run(main())
