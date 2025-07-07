#!/usr/bin/env python3
"""
Test script to demonstrate the complete four-AI integration including Claude.
This script simulates a trading opportunity to trigger the deep analysis engine.
"""

import asyncio
from datanexus_connector import DataNexus
from deep_analysis_engine import DeepAnalysisEngine

async def test_four_ai_integration():
    """
    Test the complete four-AI integration: Gemini + TimeGPT + OpenAI + Claude
    """
    print("🧪 INICIANDO PRUEBA DE INTEGRACIÓN COMPLETA DE CUATRO IAs")
    print("="*70)
    print("🤖 Sistema Multi-AI:")
    print("   1. Google Gemini (Análisis Cognitivo de Noticias)")
    print("   2. TimeGPT/Nixtla (Pronósticos Predictivos)")
    print("   3. OpenAI GPT-4 (Decisiones Finales Inteligentes)")
    print("   4. Claude/Anthropic (Contexto Macroeconómico)")
    print("="*70)
    
    # Inicializar DataNexus
    print("🔌 Inicializando DataNexus...")
    data_nexus = DataNexus()
    
    # Crear motor de análisis
    print("🧠 Inicializando Deep Analysis Engine...")
    analysis_engine = DeepAnalysisEngine(data_nexus)
    
    # Simular una oportunidad detectada por el escáner
    print("🎯 Simulando oportunidad de trading detectada...")
    test_opportunity = {
        'pair': 'BTC-USDT',
        'type': 'crypto',
        'reason': 'RSI en sobreventa (28.5) - Oportunidad de compra detectada'
    }
    
    print(f"📊 Oportunidad: {test_opportunity['pair']}")
    print(f"🔍 Trigger: {test_opportunity['reason']}")
    print("="*70)
    
    # Ejecutar análisis profundo completo
    print("🚀 Ejecutando análisis profundo con las cuatro IAs...")
    await analysis_engine.analyze_opportunity(test_opportunity)
    
    print("\n" + "="*70)
    print("✅ PRUEBA COMPLETADA")
    print("🎉 Sistema de cuatro IAs funcionando correctamente!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_four_ai_integration())