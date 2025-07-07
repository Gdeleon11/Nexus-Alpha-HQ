#!/usr/bin/env python3
"""
Deep Analysis Engine for DataNexus - Advanced Trading Opportunity Analysis
Este módulo contiene la clase principal que orquestará el análisis profundo 
de las oportunidades detectadas por el Market_Scanner.
"""

import asyncio
import os
import pandas as pd
import google.generativeai as genai
from nixtla import NixtlaClient
from openai import OpenAI
from anthropic import Anthropic
from datanexus_connector import DataNexus
import signal_storage
import requests
import json
import uuid
import sqlite3
from datetime import datetime
from telegram_notifier import send_telegram_alert
from logger import log_to_dashboard, send_signal_to_dashboard
from regime_analyzer import get_dxy_trend
from causal_engine import estimate_causal_effect
from rlhf_engine import RLHFEngine
from multimodal_analyzer import MultimodalAnalyzer
import shap
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
from counter_intel_engine import CounterIntelEngine
from explanation_engine import generate_explanation, get_explanation


class DeepAnalysisEngine:
    """
    Motor de análisis profundo que orquesta múltiples tipos de análisis
    para las oportunidades detectadas por el escáner de mercados.
    """

    def __init__(self, data_connector: DataNexus, socketio):
        """
        Inicializa el motor de análisis con una instancia de DataNexus.

        Args:
            data_connector: Instancia inicializada de DataNexus
            socketio: Instancia de SocketIO para comunicación en tiempo real
        """
        self.data_connector = data_connector
        self.socketio = socketio

        # Configurar Google Gemini API
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
                log_to_dashboard(self.socketio, "Google Gemini configurado correctamente para análisis cognitivo")
            except Exception as e:
                log_to_dashboard(self.socketio, f"Error configurando Google Gemini: {e}")
                self.llm_model = None
        else:
            log_to_dashboard(self.socketio, "GOOGLE_API_KEY no encontrada. Análisis cognitivo funcionará en modo placeholder.")
            self.llm_model = None

        # Configurar TimeGPT API
        self.timegpt_api_key = os.getenv("TIMEGPT_API_KEY")
        if self.timegpt_api_key:
            try:
                self.nixtla_client = NixtlaClient(api_key=self.timegpt_api_key)
                log_to_dashboard(self.socketio, "TimeGPT (Nixtla) configurado correctamente para pronósticos predictivos")
            except Exception as e:
                log_to_dashboard(self.socketio, f"Error configurando TimeGPT: {e}")
                self.nixtla_client = None
        else:
            log_to_dashboard(self.socketio, "TIMEGPT_API_KEY no encontrada. Pronósticos funcionarán en modo placeholder.")
            self.nixtla_client = None

        # Configurar OpenAI API para decisiones finales inteligentes
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                log_to_dashboard(self.socketio, "OpenAI GPT-4 configurado correctamente para decisiones de trading inteligentes")
            except Exception as e:
                log_to_dashboard(self.socketio, f"Error configurando OpenAI: {e}")
                self.openai_client = None
        else:
            log_to_dashboard(self.socketio, "OPENAI_API_KEY no encontrada. Decisiones funcionarán con lógica basada en reglas.")
            self.openai_client = None

        # Configurar Claude API para análisis de contexto macroeconómico
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.anthropic_api_key:
            try:
                self.claude_client = Anthropic(api_key=self.anthropic_api_key)
                log_to_dashboard(self.socketio, "Claude (Anthropic) configurado correctamente para análisis de contexto macroeconómico")
            except Exception as e:
                log_to_dashboard(self.socketio, f"Error configurando Claude: {e}")
                self.claude_client = None
        else:
            log_to_dashboard(self.socketio, "ANTHROPIC_API_KEY no encontrada. Análisis macroeconómico funcionará en modo placeholder.")
            self.claude_client = None

        # Configurar Grok API para análisis de sentimiento social
        self.xai_api_key = os.getenv("XAI_API_KEY")
        if self.xai_api_key:
            try:
                self.grok_client = OpenAI(
                    api_key=self.xai_api_key,
                    base_url="https://api.x.ai/v1"
                )
                log_to_dashboard(self.socketio, "Grok (xAI) configurado correctamente para análisis de sentimiento social")
            except Exception as e:
                log_to_dashboard(self.socketio, f"Error configurando Grok: {e}")
                self.grok_client = None
        else:
            log_to_dashboard(self.socketio, "XAI_API_KEY no encontrada. Análisis de sentimiento social funcionará en modo placeholder.")
            self.grok_client = None

        # Configurar RLHF Engine
        self.rlhf_engine = RLHFEngine(self.socketio)
        log_to_dashboard(self.socketio, "🧠 RLHF Engine inicializado para aprendizaje de patrones del operador")

        # Configurar Multimodal Analyzer
        self.multimodal_analyzer = MultimodalAnalyzer(self.socketio)
        log_to_dashboard(self.socketio, "🎥 Multimodal Analyzer inicializado para análisis de video y audio")

        # Configurar Red de Inteligencia Colectiva
        self.collective_intelligence_enabled = True
        log_to_dashboard(self.socketio, "🌐 Red de Inteligencia Colectiva activada")

        # Configurar motor XAI
        self.xai_explanations = {}  # Almacenar explicaciones por signal_id
        log_to_dashboard(self.socketio, "🔍 Motor XAI (Explicable AI) inicializado con SHAP")

        # Configurar Counter Intelligence Engine
        self.counter_intel_engine = CounterIntelEngine(self.socketio)

    async def analyze_opportunity(self, opportunity: dict):
        """
        Método principal que orquesta el análisis profundo de una oportunidad.

        Args:
            opportunity: Diccionario con información de la oportunidad
                        {'pair': 'BTC-USDT', 'type': 'crypto', 'timeframe': '5m', 'reason': 'RSI en sobreventa (22.81)'}
        """
        # 1. Imprimir mensaje de inicio
        log_to_dashboard(self.socketio, f"🧠 Iniciando análisis profundo para {opportunity['pair']}...")

        # 2. Llamar a los siete métodos internos para obtener resultados
        quantitative_result = await self._run_quantitative_analysis(opportunity['pair'])
        cognitive_result = await self._run_cognitive_analysis(opportunity)
        predictive_result = await self._run_predictive_forecast(opportunity)
        macro_context_result = await self._run_macro_context_analysis(opportunity)
        social_sentiment_result = await self._run_social_sentiment_analysis(opportunity)
        correlation_result = await self._run_correlation_analysis(opportunity['pair'])
        causal_result = await self._run_causal_analysis(opportunity)

        # 2.1. Ejecutar análisis RLHF para obtener perfil del operador
        rlhf_result = await self._run_rlhf_analysis(opportunity)

        # 2.2. Ejecutar análisis multimodal para contenido de video/audio relevante
        multimodal_result = await self._run_multimodal_analysis(opportunity)

        # 3. Agrupar todos los resultados para la decisión final
        analysis_results = {
            'trigger': opportunity['reason'],
            'cognitive': cognitive_result,
            'predictive': predictive_result,
            'macro_context': macro_context_result,
            'social_sentiment': social_sentiment_result,
            'correlation': correlation_result,
            'causal': causal_result,
            'rlhf': rlhf_result,
            'multimodal': multimodal_result
        }

        # 4. Tomar decisión final basada en confluencia
        final_decision = self._make_final_decision(analysis_results)

        # 4.1. Generar explicación XAI para la decisión
        signal_id = str(uuid.uuid4())
        explanation_data = generate_explanation(analysis_results, final_decision, signal_id)

        # 5. Mostrar la Tarjeta de Señal de Trading con decisión final
        self._display_signal_card(opportunity, quantitative_result, cognitive_result, predictive_result, macro_context_result, social_sentiment_result, correlation_result, causal_result, rlhf_result, multimodal_result, final_decision, signal_id)

        # 8. Análisis Multimodal (Video/Audio)
        multimodal_result = await self._run_multimodal_analysis(opportunity)
        analysis_results['multimodal'] = multimodal_result

        # 9. Análisis de Contrainteligencia (Detección de Manipulación)
        manipulation_score = await self._run_counter_intelligence_analysis(opportunity)
        analysis_results['manipulation_score'] = manipulation_score

    async def _run_quantitative_analysis(self, pair: str):
        """
        Análisis cuantitativo placeholder.

        Args:
            pair: Par de trading a analizar

        Returns:
            Resultado del análisis cuantitativo
        """
        return "Análisis Cuantitativo: Velas de reversión detectadas en soporte."

    async def _run_cognitive_analysis(self, opportunity: dict):
        """
        Análisis cognitivo usando Google Gemini para evaluar sentimiento de noticias.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis cognitivo basado en sentimiento de noticias
        """
        # Si Gemini no está configurado, usar fallback
        if not self.llm_model:
            return "Análisis Cognitivo: Google Gemini no configurado, análisis en modo placeholder."

        try:
            # Extraer el par y determinar el tema de búsqueda
            pair = opportunity['pair']

            # Determinar el tema para la búsqueda de noticias
            if opportunity['type'] == 'crypto':
                # Para crypto, extraer el símbolo base (ej: BTC-USDT -> BTC)
                topic = pair.split('-')[0] if '-' in pair else pair.split('USDT')[0]
            elif opportunity['type'] == 'forex':
                # Para forex, extraer las divisas (ej: OANDA:EUR_USD -> EUR)
                if 'OANDA:' in pair:
                    forex_pair = pair.replace('OANDA:', '')
                    topic = forex_pair.split('_')[0] if '_' in forex_pair else forex_pair[:3]
                else:
                    topic = pair[:3]
            else:
                topic = pair

            # Obtener noticias relacionadas al tema usando la función mejorada
            category = 'crypto' if opportunity['type'] == 'crypto' else 'forex'
            news_data = await self.data_connector.get_market_news(category=category, topic=topic, limit=5)

            # Si no hay noticias, devolver mensaje apropiado
            if not news_data or len(news_data) == 0:
                return f"Análisis Cognitivo: No hay noticias recientes disponibles para {topic}."

            # Concatenar los titulares de las noticias
            headlines = []
            for article in news_data:
                if 'headline' in article and article['headline']:
                    headlines.append(article['headline'])

            if not headlines:
                return f"Análisis Cognitivo: No se encontraron titulares relevantes para {topic}."

            lista_de_titulares = '\n'.join([f"- {headline}" for headline in headlines])

            # Crear prompt específico para Gemini
            prompt = f"""
Actúa como un analista financiero experto en sentimiento de mercado. Tu única tarea es analizar los siguientes titulares de noticias para el activo {opportunity['pair']}.

Evalúa el sentimiento general y resúmelo en una sola frase concisa. El sentimiento debe ser uno de: 'Muy Positivo', 'Positivo', 'Neutral', 'Negativo', o 'Muy Negativo'.

Titulares:
{lista_de_titulares}
"""

            # Enviar prompt a Gemini
            response = await self._generate_content_async(prompt)

            if response:
                return f"Análisis Cognitivo (Gemini): {response.strip()}"
            else:
                return f"Análisis Cognitivo: Error procesando noticias para {topic}."

        except Exception as e:
            log_to_dashboard(f"Error en análisis cognitivo con Gemini: {e}")
            return f"Análisis Cognitivo: Error al procesar - {str(e)[:50]}..."

    async def _generate_content_async(self, prompt: str):
        """
        Genera contenido usando Gemini de forma asíncrona.

        Args:
            prompt: Prompt para enviar a Gemini

        Returns:
            Respuesta de texto de Gemini o None si hay error
        """
        try:
            # Gemini no tiene un método async nativo, así que usamos asyncio
            import asyncio
            loop = asyncio.get_event_loop()

            # Ejecutar la llamada síncrona en un hilo separado
            response = await loop.run_in_executor(
                None, 
                lambda: self.llm_model.generate_content(prompt)
            )

            return response.text if response and hasattr(response, 'text') else None

        except Exception as e:
            log_to_dashboard(f"Error generando contenido con Gemini: {e}")
            return None

    async def _run_predictive_forecast(self, opportunity: dict):
        """
        Pronóstico predictivo usando TimeGPT API con Nixtla SDK.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Diccionario con predicción y confianza
        """
        # Si TimeGPT no está configurado, usar fallback
        if not self.nixtla_client:
            return {'prediction': 'Plana', 'confidence': 0.50}

        try:
            # 1. Obtener datos históricos más largos para el pronóstico
            pair = opportunity['pair']
            asset_type = opportunity['type']

            # Obtener al menos 100 velas de datos históricos
            historical_data = None
            if asset_type == 'crypto':
                historical_data = await self.data_connector.get_crypto_data(pair, interval='1h', limit=100)
            elif asset_type == 'forex':
                historical_data = await self.data_connector.get_forex_data(pair, resolution='60', limit=100)

            if historical_data is None or len(historical_data) < 50:
                log_to_dashboard(self.socketio, f"Datos insuficientes para pronóstico TimeGPT de {pair}")
                return {'prediction': 'Plana', 'confidence': 0.50}

            # 2. Formatear datos para TimeGPT usando el formato correcto de Nixtla
            # TimeGPT espera un DataFrame con columnas 'ds' (timestamp) y 'y' (valores)
            df_timegpt = pd.DataFrame({
                'ds': historical_data['timestamp'].values,
                'y': historical_data['close'].values,
                'unique_id': [pair] * len(historical_data)  # Identificador único para la serie
            })

            # Asegurar que 'ds' sea datetime
            df_timegpt['ds'] = pd.to_datetime(df_timegpt['ds'])

            current_price = df_timegpt['y'].iloc[-1]

            # 3. Realizar pronóstico con TimeGPT usando el SDK oficial
            # Ejecutar en un hilo separado ya que el SDK de Nixtla es síncrono
            loop = asyncio.get_event_loop()
            forecast_df = await loop.run_in_executor(
                None,
                lambda: self.nixtla_client.forecast(
                    df=df_timegpt,
                    h=5,  # Forecast Horizon: predecir las siguientes 5 velas
                    level=[80, 90],  # Intervalos de confianza
                    time_col='ds',
                    target_col='y',
                    id_col='unique_id'
                )
            )

            # 4. Procesar la respuesta
            if forecast_df is None or len(forecast_df) == 0:
                log_to_dashboard(self.socketio, "No se pudo obtener pronóstico de TimeGPT")
                return {'prediction': 'Plana', 'confidence': 0.50}

            # Extraer las predicciones de TimeGPT
            if 'TimeGPT' not in forecast_df.columns:
                log_to_dashboard(self.socketio, "Respuesta de TimeGPT no contiene columna 'TimeGPT'")
                return {'prediction': 'Plana', 'confidence': 0.50}

            forecast_values = forecast_df['TimeGPT'].tolist()
            avg_forecast = sum(forecast_values) / len(forecast_values)

            # 5. Generar conclusión basada en la predicción
            price_change_percent = ((avg_forecast - current_price) / current_price) * 100

            if price_change_percent > 2:
                prediction = 'Alcista'
                confidence = min(0.95, 0.50 + abs(price_change_percent) * 0.05)
            elif price_change_percent < -2:
                prediction = 'Bajista'
                confidence = min(0.95, 0.50 + abs(price_change_percent) * 0.05)
            else:
                prediction = 'Plana'
                confidence = 0.60

            log_to_dashboard(self.socketio, f"TimeGPT predicción para {pair}: {prediction} (cambio: {price_change_percent:.2f}%)")

            return {
                'prediction': prediction,
                'confidence': round(confidence, 2)
            }

        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'request limit' in error_msg:
                log_to_dashboard(self.socketio, f"TimeGPT: Límite mensual alcanzado. Contactar ops@nixtla.io para aumentar límite.")
                return {'prediction': 'Plana', 'confidence': 0.50}
            else:
                log_to_dashboard(self.socketio, f"Error en pronóstico TimeGPT: {e}")
                return {'prediction': 'Plana', 'confidence': 0.50}

    async def _run_macro_context_analysis(self, opportunity: dict):
        """
        Análisis de contexto macroeconómico usando Claude con persona de economista del FMI.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis macroeconómico
        """
        # Extraer el activo base del par de trading
        pair = opportunity['pair']
        base_asset = self._extract_base_asset(pair)

        # Generar contexto macroeconómico simulado (en una implementación real, esto se obtendría de fuentes de datos económicos)
        macro_context = self._generate_macro_context(base_asset)

        if not self.claude_client:
            log_to_dashboard(self.socketio, "Claude no configurado, usando análisis macroeconómico placeholder")
            return f"Contexto macroeconómico placeholder para {base_asset}: Neutral"

        try:
            # Prompt para Claude con persona de economista del FMI
            prompt = f"""Eres un economista senior del FMI. Basado en el siguiente texto, resume el panorama macroeconómico para el par {pair} en una frase. El sentimiento general para el activo base es Positivo, Negativo o Neutral.

Contexto macroeconómico:
{macro_context}

Responde únicamente con el formato: "Sentimiento: [Positivo/Negativo/Neutral] - [Breve justificación en una frase]"
"""

            # Llamar a Claude usando la API de messages
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extraer el contenido de la respuesta
            if response.content and len(response.content) > 0:
                # Buscar el contenido de texto en la respuesta
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        claude_response = content_block.text.strip()
                        log_to_dashboard(self.socketio, f"Claude análisis macroeconómico para {pair}: {claude_response}")
                        return claude_response

                log_to_dashboard(self.socketio, "Claude no devolvió contenido de texto válido")
                return f"Sentimiento: Neutral - Análisis no disponible"
            else:
                log_to_dashboard(self.socketio, "Claude no devolvió respuesta válida")
                return f"Sentimiento: Neutral - Análisis no disponible"

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis macroeconómico Claude: {e}")
            return f"Sentimiento: Neutral - Error en análisis ({str(e)[:50]}...)"

    async def _run_social_sentiment_analysis(self, opportunity: dict):
        """
        Análisis de sentimiento social usando Grok para evaluar el pulso en X (Twitter).

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis de sentimiento social
        """
        pair = opportunity['pair']
        base_asset = self._extract_base_asset(pair)

        if not self.grok_client:
            log_to_dashboard(self.socketio, "Grok no configurado, usando análisis de sentimiento social placeholder")
            return "Pulso Social: Neutral - Grok no disponible"

        try:
            # Prompt específico para Grok con su personalidad característica
            prompt = f"""Eres Grok. Sin rodeos y con un toque de sarcasmo, analiza el sentimiento actual en la red social X para el activo {base_asset}. Dime si el 'rebaño' está Eufórico, Optimista, Neutral, Pesimista o en Pánico. Sé directo y resume en una frase."""

            # Llamar a la API de Grok usando el cliente OpenAI compatible
            response = self.grok_client.chat.completions.create(
                model="grok-2-1212",  # Modelo de texto más robusto
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )

            if response.choices and len(response.choices) > 0:
                grok_response = response.choices[0].message.content.strip()
                log_to_dashboard(self.socketio, f"Grok análisis social para {pair}: {grok_response}")
                return f"Pulso Social: {grok_response}"
            else:
                log_to_dashboard(self.socketio, "Grok no devolvió respuesta válida")
                return f"Pulso Social: Neutral - Sin datos sociales disponibles"

        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'rate limit' in error_msg:
                log_to_dashboard(self.socketio, f"Grok: Límite de velocidad alcanzado. Reintentando más tarde.")
                return f"Pulso Social: Neutral - Límite temporal"
            else:
                log_to_dashboard(self.socketio, f"Error en análisis social Grok: {e}")
                return f"Pulso Social: Neutral - Error en análisis ({str(e)[:30]}...)"

    async def _run_correlation_analysis(self, pair: str):
        """
        Análisis de correlación usando el DXY para pares que incluyen USD.

        Args:
            pair: Par de trading a analizar

        Returns:
            Resultado del análisis de correlación
        """
        try:
            # Verificar si el par incluye USD
            if 'USD' in pair.upper():
                log_to_dashboard(self.socketio, f"Analizando correlación DXY para {pair}...")
                correlation_result = await get_dxy_trend()
                log_to_dashboard(self.socketio, f"Correlación DXY para {pair}: {correlation_result}")
                return correlation_result
            else:
                # Para pares sin USD, análisis de correlación genérico
                return f"Correlación: Par {pair} no incluye USD - Análisis de correlación estándar aplicado"

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis de correlación: {e}")
            return f"Correlación: Error en análisis - Impacto neutral asumido"

    async def _run_causal_analysis(self, opportunity: dict):
        """
        Análisis de inferencia causal usando el motor de causalidad.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis causal con score de confianza
        """
        try:
            pair = opportunity['pair']
            asset_type = opportunity['type']

            log_to_dashboard(self.socketio, f"🔬 Iniciando análisis causal para {pair}...")

            # Obtener datos históricos para el análisis causal
            historical_data = None
            if asset_type == 'crypto':
                historical_data = await self.data_connector.get_crypto_data(pair, interval='1h', limit=50)
            elif asset_type == 'forex':
                historical_data = await self.data_connector.get_forex_data(pair, resolution='60', limit=50)

            if historical_data is None or len(historical_data) < 20:
                log_to_dashboard(self.socketio, f"⚠️ Datos insuficientes para análisis causal de {pair}")
                return "Análisis Causal: Datos insuficientes - Score: 0.10"

            # Crear evento sintético basado en el trigger de la oportunidad
            trigger_reason = opportunity.get('reason', '')

            # Determinar sentimiento del evento basado en el trigger
            event_sentiment = 'neutral'
            if 'sobreventa' in trigger_reason.lower() or 'oversold' in trigger_reason.lower():
                event_sentiment = 'positive'  # Sobreventa puede indicar oportunidad de compra
            elif 'sobrecompra' in trigger_reason.lower() or 'overbought' in trigger_reason.lower():
                event_sentiment = 'negative'  # Sobrecompra puede indicar oportunidad de venta

            # Determinar intensidad del evento basado en el valor RSI si está disponible
            event_intensity = 0.5  # Default
            if 'RSI' in trigger_reason:
                try:
                    # Extraer valor RSI del mensaje
                    import re
                    rsi_match = re.search(r'RSI.*?(\d+\.?\d*)', trigger_reason)
                    if rsi_match:
                        rsi_value = float(rsi_match.group(1))
                        # Convertir RSI a intensidad (más extremo = más intensidad)
                        if rsi_value <= 30:
                            event_intensity = 0.8  # Alta intensidad para RSI muy bajo
                        elif rsi_value >= 70:
                            event_intensity = 0.8  # Alta intensidad para RSI muy alto
                        else:
                            event_intensity = 0.4  # Baja intensidad para RSI normal
                except:
                    event_intensity = 0.5

            # Usar timestamp del punto medio de los datos como momento del evento
            event_timestamp = historical_data['timestamp'].iloc[len(historical_data)//2]

            # Crear datos del evento para el análisis causal
            event_data = {
                'type': 'market_signal',
                'sentiment': event_sentiment,
                'timestamp': event_timestamp,
                'intensity': event_intensity,
                'description': trigger_reason
            }

            # Ejecutar análisis causal
            causal_score = estimate_causal_effect(event_data, historical_data)

            # Interpretar el score causal
            if causal_score >= 0.7:
                causal_interpretation = "Alta causalidad detectada"
            elif causal_score >= 0.4:
                causal_interpretation = "Causalidad moderada detectada"
            elif causal_score >= 0.2:
                causal_interpretation = "Baja causalidad detectada"
            else:
                causal_interpretation = "Causalidad insignificante"

            result = f"Análisis Causal: {causal_interpretation} - Score: {causal_score:.3f}"

            log_to_dashboard(self.socketio, f"🔬 {result}")

            return result

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis causal: {e}")
            return f"Análisis Causal: Error en procesamiento - Score: 0.10"

    async def _run_rlhf_analysis(self, opportunity: dict):
        """
        Análisis RLHF para obtener perfil de confianza del operador.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis RLHF con perfil de confianza
        """
        try:
            pair = opportunity['pair']

            log_to_dashboard(self.socketio, f"🧠 Iniciando análisis RLHF para {pair}...")

            # Actualizar perfil de confianza del operador
            confidence_profile = self.rlhf_engine.analyze_override_patterns()

            # Obtener ajuste de confianza específico para esta señal
            signal_confidence = self.rlhf_engine.get_signal_confidence_adjustment(opportunity)

            # Generar insight para el prompt de OpenAI
            rlhf_insight = self.rlhf_engine.generate_rlhf_insight(opportunity)

            # Preparar resultado estructurado
            result = {
                'confidence_profile': confidence_profile,
                'signal_confidence': signal_confidence,
                'insight': rlhf_insight,
                'total_overrides': confidence_profile['total_overrides_analyzed'],
                'dominant_pattern': confidence_profile['dominant_override_pattern']['pattern']
            }

            log_to_dashboard(self.socketio, f"🧠 RLHF: Confianza del operador para {pair}: {signal_confidence:.3f}")
            log_to_dashboard(self.socketio, f"🧠 RLHF: {rlhf_insight}")

            return f"Perfil RLHF: Confianza del operador {signal_confidence:.3f} - {result['dominant_pattern'].replace('_', ' ').title()}"

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis RLHF: {e}")
            return f"Análisis RLHF: Error en procesamiento - Usando confianza por defecto (0.8)"

    async def _run_multimodal_analysis(self, opportunity: dict):
        """
        Análisis multimodal de video y audio relacionado con el activo.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Resultado del análisis multimodal
        """
        try:
            pair = opportunity['pair']
            base_asset = self._extract_base_asset(pair)

            log_to_dashboard(self.socketio, f"🎥 Iniciando análisis multimodal para {pair}...")

            # Crear palabras clave de búsqueda basadas en el activo
            search_keywords = self._create_multimodal_search_keywords(base_asset, opportunity['type'])

            if not search_keywords:
                return "Análisis Multimodal: No se generaron palabras clave relevantes"

            log_to_dashboard(self.socketio, f"🔍 Buscando contenido multimodal: '{search_keywords}'")

            # Buscar y analizar contenido de video relevante
            multimodal_result = await self.multimodal_analyzer.analyze_financial_video_by_keywords(
                search_keywords, max_results=1
            )

            log_to_dashboard(self.socketio, f"🎥 Análisis multimodal completado para {pair}")

            return multimodal_result

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis multimodal: {e}")
            return f"Análisis Multimodal: Error en procesamiento - Contenido no disponible"

    async def _run_counter_intelligence_analysis(self, opportunity: dict):
        """
        Análisis de contrainteligencia para detectar manipulación del mercado.

        Args:
            opportunity: Diccionario con información de la oportunidad

        Returns:
            Score de manipulación (0 a 1)
        """
        try:
            pair = opportunity['pair']
            base_asset = self._extract_base_asset(pair)

            log_to_dashboard(self.socketio, f"🕵️ Iniciando análisis de contrainteligencia para {pair}...")

            # Llamar al motor de contrainteligencia
            manipulation_score = await self.counter_intel_engine.analyze_social_media(base_asset)

            log_to_dashboard(self.socketio, f"🕵️ Análisis de contrainteligencia completado para {pair}: {manipulation_score:.3f}")

            return manipulation_score

        except Exception as e:
            log_to_dashboard(self.socketio, f"Error en análisis de contrainteligencia: {e}")
            return 0.5  # Retornar un valor neutral en caso de error

    def _create_multimodal_search_keywords(self, base_asset: str, asset_type: str) -> str:
        """
        Crea palabras clave de búsqueda para contenido multimodal relevante.

        Args:
            base_asset: Activo base (BTC, EUR, etc.)
            asset_type: Tipo de activo (crypto, forex)

        Returns:
            str: Palabras clave para búsqueda de videos
        """
        keywords_map = {
            # Criptomonedas principales
            'BTC': 'Bitcoin news today analysis',
            'ETH': 'Ethereum news today analysis',
            'SOL': 'Solana news today analysis',
            'ADA': 'Cardano news today analysis',
            'DOT': 'Polkadot news today analysis',
            'LINK': 'Chainlink news today analysis',
            'AVAX': 'Avalanche news today analysis',
            'ATOM': 'Cosmos news today analysis',
            'NEAR': 'Near Protocol news today analysis',
            'UNI': 'Uniswap news today analysis',
            'DOGE': 'Dogecoin news today analysis',
            'XRP': 'Ripple XRP news today analysis',
            'LTC': 'Litecoin news today analysis',
            'ALGO': 'Algorand news today analysis',
            'VET': 'VeChain news today analysis',
            'MANA': 'Decentraland MANA news today analysis',
            'SAND': 'Sandbox SAND news today analysis',
            'SHIB': 'Shiba Inu news today analysis',

            # Divisas principales
            'EUR': 'EUR USD Euro news today ECB',
            'GBP': 'GBP USD British Pound news today Bank of England',
            'JPY': 'USD JPY Japanese Yen news today Bank of Japan',
            'CHF': 'USD CHF Swiss Franc news today SNB',
            'CAD': 'USD CAD Canadian Dollar news today Bank of Canada',
            'AUD': 'AUD USD Australian Dollar news today RBA',
            'NZD': 'NZD USD New Zealand Dollar news today RBNZ',
        }

        # Obtener palabras clave específicas del activo
        specific_keywords = keywords_map.get(base_asset.upper())

        if specific_keywords:
            return specific_keywords
        else:
            # Fallback genérico basado en tipo de activo
            if asset_type == 'crypto':
                return f"{base_asset} cryptocurrency news today analysis"
            elif asset_type == 'forex':
                return f"{base_asset} currency news today central bank"
            else:
                return f"{base_asset} financial news today analysis"

    def _extract_base_asset(self, pair: str) -> str:
        """
        Extrae el activo base del par de trading.

        Args:
            pair: Par de trading (e.g., 'BTC-USDT', 'OANDA:EUR_USD')

        Returns:
            Activo base extraído
        """
        # Manejar diferentes formatos de pares
        if 'OANDA:' in pair:
            # Formato forex: OANDA:EUR_USD -> EUR
            forex_pair = pair.replace('OANDA:', '')
            return forex_pair.split('_')[0]
        elif '-' in pair:
            # Formato crypto: BTC-USDT -> BTC
            return pair.split('-')[0]
        else:
            # Formato alternativo: BTCUSDT -> BTC (asumiendo primeros 3 caracteres)
            return pair[:3]

    def _generate_macro_context(self, base_asset: str) -> str:
        """
        Genera contexto macroeconómico simulado para el activo base.

        Args:
            base_asset: Activo base (e.g., 'BTC', 'EUR', 'GBP')

        Returns:
            Contexto macroeconómico simulado
        """
        # Contextos macroeconómicos predefinidos por tipo de activo
        if base_asset in ['BTC', 'ETH', 'ADA', 'SOL']:
            return f"""
El sector de criptomonedas está experimentando volatilidad moderada debido a las expectativas de regulación en EE.UU.
La adopción institucional de {base_asset} continúa creciendo, con mayor interés de fondos de inversión.
Las políticas monetarias restrictivas de los bancos centrales están afectando los activos de riesgo.
Los indicadores técnicos sugieren consolidación en el rango actual.
"""
        elif base_asset in ['EUR', 'GBP', 'JPY', 'AUD']:
            return f"""
El {base_asset} está influenciado por las decisiones de política monetaria del banco central correspondiente.
Las tensiones geopolíticas y la incertidumbre económica global están afectando las monedas de refugio.
Los datos de inflación y empleo recientes muestran signos de estabilización.
Las expectativas de tipos de interés están impactando los flujos de capital hacia esta divisa.
"""
        else:
            return f"""
El contexto macroeconómico para {base_asset} muestra condiciones mixtas en los mercados globales.
Las políticas fiscales y monetarias están generando incertidumbre en los mercados financieros.
Los indicadores económicos sugieren un entorno de cautela y consolidación.
"""

    def _make_final_decision(self, analysis_results: dict) -> str:
        """
        Método de decisión final usando OpenAI GPT-4 para análisis inteligente.

        Args:
            analysis_results: Diccionario con todos los resultados del análisis

        Returns:
            Veredicto final de trading
        """
        # Si OpenAI no está configurado, usar lógica de reglas como fallback
        if not self.openai_client:
            return self._fallback_decision_logic(analysis_results)

        try:
            # Construir el Prompt Maestro para GPT-4
            prompt = f"""
Eres 'Nexus-Decide', un gestor de riesgo y estratega de trading de élite. Tu única función es analizar la siguiente información de tus analistas especializados y emitir un veredicto final.

**REGLA CRÍTICA:** Solo debes recomendar 'COMPRAR' o 'VENDER' si existe una fuerte confluencia de, al menos, 6 de los 9 factores. Con la incorporación del análisis multimodal y RLHF, ahora tienes nueve dimensiones de análisis avanzado que incluyen contenido de video/audio y el perfil de confianza del operador humano. Si hay señales mixtas o débiles, tu veredicto debe ser 'NO OPERAR'.

**DATOS DE ANÁLISIS (9 DIMENSIONES):**
- **Disparador Inicial:** {analysis_results['trigger']}
- **Análisis Cognitivo (Noticias):** {analysis_results['cognitive']}
- **Análisis Predictivo (Forecast):** Predicción {analysis_results['predictive']['prediction']} con {analysis_results['predictive']['confidence']:.0%} de confianza.
- **Contexto Macroeconómico (Claude):** {analysis_results['macro_context']}
- **Pulso Social (Grok):** {analysis_results['social_sentiment']}
- **Análisis de Correlación:** {analysis_results['correlation']}
- **Análisis Causal:** {analysis_results['causal']}
- **Perfil RLHF del Operador:** {analysis_results['rlhf']}
- **Análisis Multimodal (Video/Audio):** {analysis_results['multimodal']}

**PESO ESPECIAL DEL ANÁLISIS MULTIMODAL:** 
El análisis multimodal tiene un peso crítico ya que captura señales no verbales de ejecutivos, analistas y presentadores financieros. Cuando el análisis multimodal detecta incongruencias entre lo que se dice y cómo se dice (lenguaje corporal vs. palabras), esto es una señal de ALERTA MÁXIMA que debe influir fuertemente en tu decisión.

**FACTORES DE EVASIÓN MULTIMODAL:**
- Si el análisis multimodal detecta "evasión", "duda" o "nerviosismo" → Reducir confianza en señales positivas
- Si detecta "confianza genuina" y "coherencia multimodal" → Aumentar peso de señales positivas
- Las incongruencias multimodales pueden invalidar hasta 3 factores positivos

Basado estrictamente en estos nueve datos y en tu regla crítica reforzada, emite tu veredicto final. Tu respuesta debe ser una única palabra: COMPRAR, VENDER, o NO OPERAR.
"""

            # Llamar a la API de OpenAI GPT-4
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3  # Baja temperatura para decisiones consistentes
            )

            # Procesar y validar la respuesta
            decision = response.choices[0].message.content.strip().upper()

            # Validar que la respuesta sea una de las opciones válidas
            valid_decisions = ["COMPRAR", "VENDER", "NO OPERAR"]
            if decision in valid_decisions:
                log_to_dashboard(self.socketio, f"Nexus-Decide (GPT-4): {decision}")
                return decision
            else:
                log_to_dashboard(self.socketio, f"Respuesta inválida de GPT-4: {decision}. Usando fallback.")
                return self._fallback_decision_logic(analysis_results)

        except Exception as e:
            error_msg = str(e)
            if 'insufficient_quota' in error_msg or '429' in error_msg:
                log_to_dashboard(self.socketio, "GPT-4: Cuota insuficiente. Verificar facturación en OpenAI. Usando lógica de reglas.")
            else:
                log_to_dashboard(self.socketio, f"Error en decisión GPT-4: {e}. Usando fallback.")
            return self._fallback_decision_logic(analysis_results)

    def _fallback_decision_logic(self, analysis_results: dict) -> str:
        """
        Lógica de decisión basada en reglas como fallback cuando OpenAI no está disponible.

        Args:
            analysis_results: Diccionario con todos los resultados del análisis

        Returns:
            Veredicto final de trading
        """
        trigger = analysis_results.get('trigger', '')
        cognitive = analysis_results.get('cognitive', '')
        predictive = analysis_results.get('predictive', {}).get('prediction', '')
        confidence = analysis_results.get('predictive', {}).get('confidence', 0.5)

        # Regla para COMPRAR - RSI sobreventa + predicción alcista + sentimiento no negativo
        if 'sobreventa' in trigger and predictive == 'Alcista' and 'Negativo' not in cognitive:
            return "COMPRAR"

        # Regla para VENDER - RSI sobrecompra + predicción bajista + sentimiento no positivo
        if 'sobrecompra' in trigger and predictive == 'Bajista' and 'Positivo' not in cognitive:
            return "VENDER"

        # Reglas adicionales para señales parciales
        if 'sobreventa' in trigger and predictive == 'Alcista':
            return "COMPRAR"

        if 'sobrecompra' in trigger and predictive == 'Bajista':
            return "VENDER"

        # Si no hay confluencia clara, no operar
        return "NO OPERAR"

    def _display_signal_card(self, opportunity, quantitative_result, cognitive_result, predictive_result, macro_context_result, social_sentiment_result, correlation_result, causal_result, rlhf_result, multimodal_result, final_decision, signal_id):
        """
        Envía la Tarjeta de Señal formateada al dashboard del frontend.

        Args:
            opportunity: Información original de la oportunidad
            quantitative_result: Resultado del análisis cuantitativo
            cognitive_result: Resultado del análisis cognitivo
            predictive_result: Resultado del pronóstico predictivo
            macro_context_result: Resultado del análisis macroeconómico
            social_sentiment_result: Resultado del análisis de sentimiento social
            final_decision: Veredicto final basado en confluencia
            signal_id: ID único de la señal (ya generado)
        """

        # Calcular expiración recomendada basada en el timeframe
        timeframe = opportunity.get('timeframe', '5m')
        expiration_map = {
            '1m': '2 Minutos',
            '5m': '10 Minutos', 
            '15m': '30 Minutos',
            '30m': '1 Hora',
            '1h': '2 Horas',
            '4h': '8 Horas',
            '1d': '1 Día'
        }
        recommended_expiration = expiration_map.get(timeframe, '10 Minutos')

        # Preparar datos estructurados para el dashboard compatible con frontend
        signal_card_data = {
            'signal_id': signal_id,
            'pair': opportunity['pair'],
            'type': opportunity['type'].upper(),
            'timeframe': timeframe,
            'trigger': opportunity['reason'],
            'cognitive': cognitive_result,
            'predictive': {
                'prediction': predictive_result['prediction'],
                'confidence': predictive_result['confidence']
            },
            'macro': macro_context_result,
            'social': social_sentiment_result,
            'correlation': correlation_result,
            'causal': causal_result,
            'rlhf': rlhf_result,
            'multimodal': multimodal_result,
            'verdict': final_decision,
            'expiration': recommended_expiration,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Enviar tarjeta de señal al dashboard
        send_signal_to_dashboard(self.socketio, signal_card_data)

        # También enviar log básico
        log_to_dashboard(self.socketio, f"🎯 SEÑAL GENERADA: {opportunity['pair']} - {final_decision}")

        # Store signal if it's a BUY or SELL decision
        if "COMPRAR" in final_decision or "VENDER" in final_decision:
            try:
                # Registrar señal en la base de datos con estado PENDING
                conn = sqlite3.connect('trades.db')
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO trades (signal_id, timestamp, asset, signal_type, verdict, result)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    datetime.now().isoformat(),
                    opportunity['pair'],
                    'BUY' if 'COMPRAR' in final_decision else 'SELL',
                    final_decision,
                    'PENDING'
                ))

                conn.commit()
                conn.close()

                log_to_dashboard(self.socketio, f"✅ SEÑAL REGISTRADA EN BD [ID: {signal_id[:8]}...]. Lista para tracking manual.")

                # Registrar hook para envío a Red de Inteligencia Colectiva cuando se actualice el resultado
                self._register_collective_intelligence_hook(signal_id, opportunity['pair'], timeframe, final_decision)

                # Get current market price to store with signal
                import asyncio
                data = asyncio.run(self.data_connector.get_crypto_data(opportunity['pair'], '1m', 1))
                current_price = data['close'].iloc[-1] if data is not None and not data.empty else None

                signal_data = {
                    'pair': opportunity['pair'],
                    'type': opportunity['type'],
                    'action': 'BUY' if 'COMPRAR' in final_decision else 'SELL',
                    'price': current_price,
                    'trigger': opportunity['reason'],
                    'quantitative': quantitative_result,
                    'cognitive': cognitive_result,
                    'predictive': f"{predictive_result['prediction']} ({predictive_result['confidence']:.0%})",
                    'macro_context': macro_context_result,
                    'social_sentiment': social_sentiment_result,
                    'correlation': correlation_result,
                    'causal': causal_result,
                    'rlhf': rlhf_result,
                    'decision': final_decision
                }

                # Mantener compatibilidad con el sistema provisional existente
                stored_signal_id = signal_storage.store_signal(signal_data)

                # Enviar notificación de Telegram para señales de COMPRAR o VENDER
                telegram_message = self._format_telegram_message(
                    opportunity, quantitative_result, cognitive_result, 
                    predictive_result, macro_context_result, social_sentiment_result, 
                    correlation_result, causal_result, rlhf_result, final_decision, signal_id, current_price
                )

                try:
                    telegram_sent = send_telegram_alert(telegram_message)
                    if telegram_sent:
                        log_to_dashboard(self.socketio, "📱 Alerta de Telegram enviada exitosamente")
                    else:
                        log_to_dashboard(self.socketio, "⚠️  No se pudo enviar alerta de Telegram")
                except Exception as telegram_error:
                    log_to_dashboard(self.socketio, f"⚠️  Error enviando Telegram: {telegram_error}")

            except Exception as e:
                log_to_dashboard(self.socketio, f"⚠️  Error al almacenar señal: {e}")

        log_to_dashboard(self.socketio, "="*60)

    def _format_telegram_message(self, opportunity, quantitative_result, cognitive_result, 
                                predictive_result, macro_context_result, social_sentiment_result, 
                                correlation_result, causal_result, rlhf_result, final_decision, signal_id, current_price):
        """
        Formatea el mensaje de Telegram para alertas de trading.

        Args:
            opportunity: Información de la oportunidad
            quantitative_result: Análisis cuantitativo
            cognitive_result: Análisis cognitivo
            predictive_result: Resultado predictivo
            macro_context_result: Análisis macroeconómico
            social_sentiment_result: Análisis de sentimiento social
            final_decision: Decisión final
            signal_id: ID de la señal
            current_price: Precio actual

        Returns:
            Mensaje formateado para Telegram
        """
        # Determinar emoji según la decisión
        action_emoji = "🟢" if "COMPRAR" in final_decision else "🔴"
        action_text = "COMPRAR" if "COMPRAR" in final_decision else "VENDER"

        # Formatear precio si está disponible
        price_text = f"${current_price:.2f}" if current_price else "N/A"

        # Get timeframe information
        timeframe = opportunity.get('timeframe', '5m')
        expiration_map = {
            '1m': '2 Minutos',
            '5m': '10 Minutos', 
            '15m': '30 Minutos',
            '30m': '1 Hora',
            '1h': '2 Horas',
            '4h': '8 Horas',
            '1d': '1 Día'
        }
        recommended_expiration = expiration_map.get(timeframe, '10 Minutos')

        message = f"""🎯 *SEÑAL DE TRADING* {action_emoji}

📊 *Par:* {opportunity['pair']}
🔍 *Tipo:* {opportunity['type'].upper()}
⏰ *Timeframe:* {timeframe.upper()}
⌛ *Expiración:* {recommended_expiration}
⚡ *Trigger:* {opportunity['reason']}
💰 *Precio Actual:* {price_text}

📈 *ANÁLISIS MULTI-AI:*
🔢 *Cuantitativo:* {quantitative_result}
🧠 *Cognitivo:* {cognitive_result}
🔮 *Predicción:* {predictive_result['prediction']} ({predictive_result['confidence']:.0%})
🌍 *Macro:* {macro_context_result}
📱 *Social:* {social_sentiment_result}
🔗 *Correlación:* {correlation_result}
🔬 *Causal:* {causal_result}
🎯 *RLHF:* {rlhf_result}

🎯 *DECISIÓN:* *{action_text}* {action_emoji}

🆔 *Signal ID:* {signal_id}
⏰ Esperando verificación del Local_Eye"""

        return message

    async def _send_to_collective_intelligence(self, signal_id: str, asset: str, timeframe: str, verdict: str, result: str):
        """
        Envía datos anonimizados a la Red de Inteligencia Colectiva.

        Args:
            signal_id: ID de la señal
            asset: Activo negociado
            timeframe: Marco temporal
            verdict: Veredicto original
            result: Resultado de la operación (WIN/LOSS)
        """
        if not self.collective_intelligence_enabled or result not in ['WIN', 'LOSS']:
            return

        try:
            # Extraer estrategia del veredicto de forma más inteligente
            estrategia_base = "RSI"  # Por defecto
            
            # Detectar tipo de estrategia del veredicto
            if "MACD" in verdict.upper():
                estrategia_base = "MACD"
            elif "SMA" in verdict.upper():
                estrategia_base = "SMA"
            elif "RSI" in verdict.upper():
                estrategia_base = "RSI"
            
            if "COMPRAR" in verdict:
                estrategia_usada = f"{estrategia_base}_BUY"
            elif "VENDER" in verdict:
                estrategia_usada = f"{estrategia_base}_SELL"
            else:
                estrategia_usada = f"{estrategia_base}_NEUTRAL"

            # Detectar régimen de mercado actual (simplificado)
            try:
                from regime_analyzer import get_market_regime
                market_regime = await get_market_regime()
                if not market_regime:
                    market_regime = "UNKNOWN"
            except:
                market_regime = "UNKNOWN"

            # Datos anonimizados expandidos
            collective_data = {
                'estrategia_usada': estrategia_usada,
                'activo': asset,
                'timeframe': timeframe,
                'resultado_win_loss': result,
                'regimen_mercado': market_regime
            }

            # Enviar a servidor central (mismo servidor, puerto 5000)
            api_url = "http://0.0.0.0:5000/api/log_collective_trade"

            response = requests.post(
                api_url,
                json=collective_data,
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                log_to_dashboard(self.socketio, f"🌐 Datos enviados a Red de Inteligencia Colectiva: {estrategia_usada}")
                log_to_dashboard(self.socketio, f"🔗 Contribuyendo al conocimiento global de la red Nexus-Alpha")
            else:
                log_to_dashboard(self.socketio, f"⚠️ Error enviando a Red de Inteligencia: {response.status_code}")

        except requests.exceptions.RequestException as e:
            log_to_dashboard(self.socketio, f"⚠️ Red de Inteligencia no disponible: {str(e)[:50]}...")
        except Exception as e:
            log_to_dashboard(self.socketio, f"⚠️ Error en Red de Inteligencia: {str(e)[:50]}...")

    def trigger_collective_intelligence_update(self, signal_id: str):
        """
        Activar envío a Red de Inteligencia cuando se actualice resultado de señal.
        """
        try:
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()
            
            # Obtener información de la señal actualizada
            cursor.execute('''
                SELECT asset, verdict, result FROM trades 
                WHERE signal_id = ? AND result IN ('WIN', 'LOSS')
            ''', (signal_id,))
            
            trade_info = cursor.fetchone()
            conn.close()
            
            if trade_info:
                asset, verdict, result = trade_info
                
                # Determinar timeframe desde el activo o veredicto
                timeframe = "5m"  # Default
                for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    if tf in str(asset) or tf in str(verdict):
                        timeframe = tf
                        break
                
                # Enviar a Red de Inteligencia de forma asíncrona
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(
                        self._send_to_collective_intelligence(signal_id, asset, timeframe, verdict, result)
                    )
                except:
                    # Si no hay loop, crear uno nuevo
                    asyncio.run(
                        self._send_to_collective_intelligence(signal_id, asset, timeframe, verdict, result)
                    )
                
        except Exception as e:
            log_to_dashboard(self.socketio, f"⚠️ Error activando inteligencia colectiva: {str(e)[:50]}...")

    def _register_collective_intelligence_hook(self, signal_id: str, asset: str, timeframe: str, verdict: str):
        """
        Registra un hook para enviar datos a la Red de Inteligencia cuando se actualice el resultado.

        Args:
            signal_id: ID de la señal
            asset: Activo
            timeframe: Marco temporal  
            verdict: Veredicto
        """
        try:
            # Almacenar información del hook en base de datos
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()

            # Crear tabla de hooks si no existe
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collective_hooks (
                    signal_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')

            cursor.execute('''
                INSERT OR REPLACE INTO collective_hooks (signal_id, asset, timeframe, verdict, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (signal_id, asset, timeframe, verdict, datetime.now().isoformat()))

            conn.commit()
            conn.close()

            log_to_dashboard(self.socketio, f"🔗 Hook registrado para Red de Inteligencia: {signal_id[:8]}...")

        except Exception as e:
            log_to_dashboard(self.socketio, f"⚠️ Error registrando hook: {str(e)[:50]}...")

    def _generate_explanation(self, analysis_results: dict, final_decision: str, signal_id: str) -> dict:
        """
        Genera explicación XAI usando SHAP para entender qué factores influenciaron la decisión.

        Args:
            analysis_results: Diccionario con todos los resultados del análisis
            final_decision: Decisión final tomada
            signal_id: ID único de la señal

        Returns:
            Diccionario con datos de explicación SHAP
        """
        try:
            log_to_dashboard(self.socketio, f"🔍 Generando explicación XAI para señal {signal_id[:8]}...")

            # Convertir análisis cualitativos a valores numéricos
            feature_values = self._convert_analysis_to_features(analysis_results)
            feature_names = list(feature_values.keys())
            values_array = np.array(list(feature_values.values())).reshape(1, -1)

            # Crear un modelo mock para SHAP basado en la lógica de decisión
            def decision_model(X):
                predictions = []
                for row in X:
                    # Simular lógica de decisión basada en los factores
                    score = 0

                    # Factor RSI (trigger)
                    if row[0] < -0.5:  # RSI sobreventa
                        score += 0.3
                    elif row[0] > 0.5:  # RSI sobrecompra
                        score -= 0.3

                    # Factor cognitivo (sentimiento)
                    score += row[1] * 0.25

                    # Factor predictivo
                    score += row[2] * 0.3

                    # Factor macroeconómico
                    score += row[3] * 0.2

                    # Factor social
                    score += row[4] * 0.15

                    # Factor correlación
                    score += row[5] * 0.1

                    # Factor causal
                    score += row[6] * 0.15

                    # Factor RLHF
                    score += row[7] * 0.2

                    # Factor multimodal
                    score += row[8] * 0.1

                    predictions.append([1-score, score])  # [NO_OPERAR_prob, OPERAR_prob]

                return np.array(predictions)

            # Crear explicador SHAP
            background_data = np.zeros((10, len(feature_names)))  # Datos de referencia
            explainer = shap.KernelExplainer(decision_model, background_data)

            # Calcular valores SHAP
            shap_values = explainer.shap_values(values_array)

            # Procesar resultados para la decisión específica
            decision_index = 1 if final_decision in ['COMPRAR', 'VENDER'] else 0
            feature_impacts = shap_values[decision_index][0]  # Primera (y única) muestra

            # Crear estructura de datos para explicación
            explanation_data = {
                'signal_id': signal_id,
                'decision': final_decision,
                'feature_names': feature_names,
                'feature_values': list(feature_values.values()),
                'feature_impacts': feature_impacts.tolist(),
                'base_value': explainer.expected_value[decision_index],
                'prediction_value': decision_model(values_array)[0][decision_index],
                'timestamp': datetime.now().isoformat()
            }

            # Almacenar explicación
            self.xai_explanations[signal_id] = explanation_data

            # Generar gráfico de fuerza usando Plotly
            plot_data = self._create_force_plot(explanation_data)
            explanation_data['plot_data'] = plot_data

            log_to_dashboard(self.socketio, f"✅ Explicación XAI generada: {len(feature_names)} factores analizados")

            return explanation_data

        except Exception as e:
            log_to_dashboard(self.socketio, f"⚠️ Error generando explicación XAI: {str(e)[:50]}...")

            # Fallback: explicación simple sin SHAP
            return {
                'signal_id': signal_id,
                'decision': final_decision,
                'feature_names': ['Trigger', 'Cognitive', 'Predictive', 'Macro', 'Social'],
                'feature_impacts': [0.2, 0.1, 0.3, 0.2, 0.2],
                'explanation': 'Explicación simplificada - Error en análisis SHAP',
                'timestamp': datetime.now().isoformat()
            }

    def _convert_analysis_to_features(self, analysis_results: dict) -> dict:
        """
        Convierte los resultados de análisis cualitativos a valores numéricos para SHAP.

        Args:
            analysis_results: Resultados del análisis

        Returns:
            Diccionario con valores numéricos
        """
        features = {}

        # Factor 1: RSI Trigger (-1 a 1)
        trigger = analysis_results.get('trigger', '')
        if 'sobreventa' in trigger.lower():
            features['RSI_Trigger'] = -0.8
        elif 'sobrecompra' in trigger.lower():
            features['RSI_Trigger'] = 0.8
        else:
            features['RSI_Trigger'] = 0.0

        # Factor 2: Sentimiento Cognitivo (-1 a 1)
        cognitive = analysis_results.get('cognitive', '')
        if 'muy positivo' in cognitive.lower():
            features['Sentiment_Cognitive'] = 1.0
        elif 'positivo' in cognitive.lower():
            features['Sentiment_Cognitive'] = 0.5
        elif 'negativo' in cognitive.lower():
            features['Sentiment_Cognitive'] = -0.5
        elif 'muy negativo' in cognitive.lower():
            features['Sentiment_Cognitive'] = -1.0
        else:
            features['Sentiment_Cognitive'] = 0.0

        # Factor 3: Predicción (-1 a 1)
        predictive = analysis_results.get('predictive', {})
        prediction = predictive.get('prediction', 'Plana')
        confidence = predictive.get('confidence', 0.5)

        if prediction == 'Alcista':
            features['Predictive_Forecast'] = confidence
        elif prediction == 'Bajista':
            features['Predictive_Forecast'] = -confidence
        else:
            features['Predictive_Forecast'] = 0.0

        # Factor 4: Macro (-1 a 1)
        macro = analysis_results.get('macro_context', '')
        if 'positivo' in macro.lower():
            features['Macro_Context'] = 0.6
        elif 'negativo' in macro.lower():
            features['Macro_Context'] = -0.6
        else:
            features['Macro_Context'] = 0.0

        # Factor 5: Social (-1 a 1)
        social = analysis_results.get('social_sentiment', '')
        if 'eufórico' in social.lower() or 'optimista' in social.lower():
            features['Social_Sentiment'] = 0.7
        elif 'pesimista' in social.lower() or 'pánico' in social.lower():
            features['Social_Sentiment'] = -0.7
        else:
            features['Social_Sentiment'] = 0.0

        # Factor 6: Correlación (-1 a 1)
        correlation = analysis_results.get('correlation', '')
        if 'favorable' in correlation.lower() or 'positiv' in correlation.lower():
            features['Market_Correlation'] = 0.4
        elif 'desfavorable' in correlation.lower() or 'negativ' in correlation.lower():
            features['Market_Correlation'] = -0.4
        else:
            features['Market_Correlation'] = 0.0

        # Factor 7: Causal (0 a 1)
        causal = analysis_results.get('causal', '')
        import re
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', causal)
        if score_match:
            causal_score = float(score_match.group(1))
            features['Causal_Analysis'] = min(causal_score, 1.0)
        else:
            features['Causal_Analysis'] = 0.1

        # Factor 8: RLHF (-1 a 1)
        rlhf = analysis_results.get('rlhf', '')
        confidence_match = re.search(r'(\d+\.?\d*)', rlhf)
        if confidence_match:
            rlhf_confidence = float(confidence_match.group(1))
            features['RLHF_Confidence'] = (rlhf_confidence - 0.5) * 2  # Normalizar a -1,1
        else:
            features['RLHF_Confidence'] = 0.0

        # Factor 9: Multimodal (-1 a 1)
        multimodal = analysis_results.get('multimodal', '')
        if 'positiv' in multimodal.lower() or 'alcista' in multimodal.lower():
            features['Multimodal_Analysis'] = 0.5
        elif 'negativ' in multimodal.lower() or 'bajista' in multimodal.lower():
            features['Multimodal_Analysis'] = -0.5
        else:
            features['Multimodal_Analysis'] = 0.0

        return features

    def _create_force_plot(self, explanation_data: dict) -> dict:
        """
        Crea un gráfico de fuerza usando Plotly para visualizar la explicación.

        Args:
            explanation_data: Datos de explicación SHAP

        Returns:
            Datos del gráfico en formato JSON
        """
        try:
            feature_names = explanation_data['feature_names']
            feature_impacts = explanation_data['feature_impacts']
            base_value = explanation_data['base_value']

            # Separar impactos positivos y negativos
            positive_impacts = [max(0, impact) for impact in feature_impacts]
            negative_impacts = [min(0, impact) for impact in feature_impacts]

            # Crear gráfico de barras horizontales
            fig = go.Figure()

            # Barras positivas (verdes)
            fig.add_trace(go.Bar(
                y=feature_names,
                x=positive_impacts,
                orientation='h',
                name='Impacto Positivo',
                marker_color='rgba(34, 139, 34, 0.8)',
                text=[f'+{impact:.3f}' if impact > 0 else '' for impact in positive_impacts],
                textposition='outside'
            ))

            # Barras negativas (rojas)
            fig.add_trace(go.Bar(
                y=feature_names,
                x=negative_impacts,
                orientation='h',
                name='Impacto Negativo',
                marker_color='rgba(220, 20, 60, 0.8)',
                text=[f'{impact:.3f}' if impact < 0 else '' for impact in negative_impacts],
                textposition='outside'
            ))

            # Configurar layout
            fig.update_layout(
                title='🔍 Explicación XAI - Factores de Decisión',
                xaxis_title='Impacto en la Decisión',
                yaxis_title='Factores de Análisis',
                barmode='relative',
                height=500,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            # Añadir línea base
            fig.add_vline(x=base_value, line_dash="dash", line_color="gray", 
                         annotation_text=f"Base: {base_value:.3f}")

            # Convertir a JSON
            plot_json = pio.to_json(fig)

            return json.loads(plot_json)

        except Exception as e:
            log_to_dashboard(self.socketio, f"⚠️ Error creando gráfico XAI: {str(e)[:50]}...")
            return {}

    def get_explanation(self, signal_id: str) -> dict:
        """
        Obtiene la explicación XAI para una señal específica.

        Args:
            signal_id: ID de la señal

        Returns:
            Datos de explicación o None si no existe
        """
        return get_explanation(signal_id)


# Función de prueba independiente
async def test_deep_analysis_engine():
    """
    Función de prueba para el motor de análisis profundo.
    """
    log_to_dashboard("🧪 Probando Deep Analysis Engine...")

    # Inicializar DataNexus
    data_nexus = DataNexus()

    # Crear motor de análisis (necesita socketio para funcionar correctamente)
    # analysis_engine = DeepAnalysisEngine(data_nexus, socketio)

    # Oportunidad de prueba
    test_opportunity = {
        'pair': 'BTC-USDT',
        'type': 'crypto',
        'reason': 'RSI en sobreventa (22.81)'
    }

    # Realizar análisis
    await analysis_engine.analyze_opportunity(test_opportunity)

    log_to_dashboard("\n✅ Prueba del Deep Analysis Engine completada.")


if __name__ == "__main__":
    asyncio.run(test_deep_analysis_engine())