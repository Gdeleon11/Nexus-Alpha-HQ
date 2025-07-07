
#!/usr/bin/env python3
"""
Counter Intelligence Engine for DataNexus
Módulo de contrainteligencia para detectar manipulación artificial del mercado
a través del análisis de patrones anómalos en redes sociales y correlaciones.
"""

import os
import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import statistics
from logger import log_to_dashboard


class CounterIntelEngine:
    """
    Motor de contrainteligencia que detecta manipulación artificial del mercado
    mediante análisis de patrones sociales y correlaciones temporales.
    """
    
    def __init__(self, socketio=None):
        """
        Inicializa el motor de contrainteligencia.
        
        Args:
            socketio: Instancia de SocketIO para logs en tiempo real
        """
        self.socketio = socketio
        
        # Configuración de APIs (simuladas para demo)
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        
        # Configuración de detección
        self.manipulation_thresholds = {
            'coordinated_posting': 0.6,  # Umbral para posting coordinado
            'hashtag_anomaly': 0.7,      # Umbral para anomalías de hashtags
            'account_age_suspicious': 30, # Días mínimos para cuenta sospechosa
            'sentiment_spike': 2.5,      # Desviaciones estándar para picos
            'temporal_correlation': 0.8   # Correlación temporal mínima
        }
        
        # Cache para análisis temporal
        self.social_history = defaultdict(list)
        self.news_history = defaultdict(list)
        
        if self.socketio:
            log_to_dashboard(self.socketio, "🕵️ Counter Intelligence Engine inicializado")
    
    async def analyze_social_media(self, asset_pair: str, timeframe_hours: int = 24) -> float:
        """
        Función principal que analiza la manipulación del mercado para un activo.
        
        Args:
            asset_pair: Símbolo del activo (ej: 'BTC-USDT', 'OANDA:EUR_USD')
            timeframe_hours: Ventana de tiempo para análisis en horas
            
        Returns:
            float: Score de manipulación (0.0 a 1.0)
        """
        try:
            # Extract base asset from pair
            asset = self._extract_base_asset(asset_pair)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"🕵️ Iniciando análisis de contrainteligencia para {asset_pair}")
            
            manipulation_scores = []
            
            # 1. Análisis de Fuente Social Media
            social_manipulation_score = await self._analyze_social_media_patterns(asset, timeframe_hours)
            manipulation_scores.append(social_manipulation_score)
            
            # 2. Correlación de Eventos
            correlation_score = await self._analyze_event_correlation(asset, timeframe_hours)
            manipulation_scores.append(correlation_score)
            
            # 3. Análisis de Actividad Coordinada
            coordination_score = await self._analyze_coordinated_activity(asset, timeframe_hours)
            manipulation_scores.append(coordination_score)
            
            # 4. Detección de Anomalías Temporales
            temporal_score = await self._analyze_temporal_anomalies(asset, timeframe_hours)
            manipulation_scores.append(temporal_score)
            
            # Calcular score final (promedio ponderado)
            weights = [0.3, 0.25, 0.25, 0.2]  # Social, Correlación, Coordinación, Temporal
            final_score = sum(score * weight for score, weight in zip(manipulation_scores, weights))
            
            # Registrar resultados
            if self.socketio:
                if final_score > 0.7:
                    log_to_dashboard(self.socketio, f"🚨 ALERTA: Manipulación detectada en {asset_pair} (Score: {final_score:.3f})")
                elif final_score > 0.5:
                    log_to_dashboard(self.socketio, f"⚠️ Actividad sospechosa en {asset_pair} (Score: {final_score:.3f})")
                else:
                    log_to_dashboard(self.socketio, f"✅ Actividad normal en {asset_pair} (Score: {final_score:.3f})")
            
            return min(final_score, 1.0)  # Asegurar que no exceda 1.0
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis de contrainteligencia para {asset_pair}: {e}")
            return 0.5  # Score neutral en caso de error
    
    async def _analyze_social_media_patterns(self, asset: str, timeframe_hours: int) -> float:
        """
        Analiza patrones anómalos en redes sociales.
        
        Args:
            asset: Símbolo del activo
            timeframe_hours: Ventana de tiempo
            
        Returns:
            float: Score de manipulación social (0.0 a 1.0)
        """
        try:
            # Simular datos de redes sociales (en implementación real, usar Twitter API)
            social_data = await self._fetch_social_media_data(asset, timeframe_hours)
            
            if not social_data:
                return 0.0
            
            manipulation_indicators = []
            
            # 1. Análisis de cuentas de baja antigüedad
            new_account_ratio = self._calculate_new_account_ratio(social_data)
            manipulation_indicators.append(min(new_account_ratio * 2, 1.0))
            
            # 2. Análisis de contenido similar/duplicado
            content_similarity = self._calculate_content_similarity(social_data)
            manipulation_indicators.append(content_similarity)
            
            # 3. Análisis de picos anómalos de hashtags
            hashtag_anomaly = self._detect_hashtag_anomalies(social_data, asset)
            manipulation_indicators.append(hashtag_anomaly)
            
            # 4. Análisis de actividad horaria
            temporal_clustering = self._analyze_temporal_clustering(social_data)
            manipulation_indicators.append(temporal_clustering)
            
            # Calcular score promedio
            social_score = statistics.mean(manipulation_indicators)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"📱 Análisis social {asset}: Score {social_score:.3f}")
            
            return social_score
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis social: {e}")
            return 0.0
    
    async def _analyze_event_correlation(self, asset: str, timeframe_hours: int) -> float:
        """
        Analiza correlación entre picos de actividad social y publicación de noticias.
        
        Args:
            asset: Símbolo del activo
            timeframe_hours: Ventana de tiempo
            
        Returns:
            float: Score de correlación sospechosa (0.0 a 1.0)
        """
        try:
            # Obtener datos de noticias y actividad social
            news_timestamps = await self._fetch_news_timestamps(asset, timeframe_hours)
            social_spikes = await self._detect_social_spikes(asset, timeframe_hours)
            
            if not news_timestamps or not social_spikes:
                return 0.0
            
            # Calcular correlaciones temporales
            correlations = []
            
            for news_time in news_timestamps:
                for spike_time in social_spikes:
                    time_diff = abs((spike_time - news_time).total_seconds() / 3600)  # Diferencia en horas
                    
                    # Si la actividad social precede a la noticia por menos de 2 horas
                    if spike_time < news_time and time_diff <= 2:
                        correlation_strength = max(0, 1 - (time_diff / 2))
                        correlations.append(correlation_strength)
            
            if not correlations:
                return 0.0
            
            # Score basado en correlaciones encontradas
            correlation_score = min(statistics.mean(correlations) * len(correlations) / 3, 1.0)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"🔗 Correlación eventos {asset}: Score {correlation_score:.3f}")
            
            return correlation_score
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en correlación de eventos: {e}")
            return 0.0
    
    async def _analyze_coordinated_activity(self, asset: str, timeframe_hours: int) -> float:
        """
        Detecta actividad coordinada entre múltiples cuentas.
        
        Args:
            asset: Símbolo del activo
            timeframe_hours: Ventana de tiempo
            
        Returns:
            float: Score de coordinación (0.0 a 1.0)
        """
        try:
            social_data = await self._fetch_social_media_data(asset, timeframe_hours)
            
            if not social_data:
                return 0.0
            
            coordination_indicators = []
            
            # 1. Análisis de timing coordinado
            post_times = [post['timestamp'] for post in social_data]
            timing_clustering = self._calculate_timing_clustering(post_times)
            coordination_indicators.append(timing_clustering)
            
            # 2. Análisis de lenguaje similar
            language_similarity = self._analyze_language_patterns(social_data)
            coordination_indicators.append(language_similarity)
            
            # 3. Análisis de red de cuentas
            network_suspicion = self._analyze_account_network(social_data)
            coordination_indicators.append(network_suspicion)
            
            coordination_score = statistics.mean(coordination_indicators)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"🤝 Actividad coordinada {asset}: Score {coordination_score:.3f}")
            
            return coordination_score
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis de coordinación: {e}")
            return 0.0
    
    async def _analyze_temporal_anomalies(self, asset: str, timeframe_hours: int) -> float:
        """
        Detecta anomalías temporales en patrones de actividad.
        
        Args:
            asset: Símbolo del activo
            timeframe_hours: Ventana de tiempo
            
        Returns:
            float: Score de anomalías temporales (0.0 a 1.0)
        """
        try:
            # Obtener histórico de actividad social
            current_activity = await self._get_current_activity_level(asset)
            historical_baseline = await self._get_historical_baseline(asset, days=7)
            
            if not historical_baseline:
                return 0.0
            
            # Calcular desviación estándar
            baseline_mean = statistics.mean(historical_baseline)
            baseline_std = statistics.stdev(historical_baseline) if len(historical_baseline) > 1 else 0
            
            if baseline_std == 0:
                return 0.0
            
            # Calcular Z-score (cuántas desviaciones estándar está el actual)
            z_score = abs(current_activity - baseline_mean) / baseline_std
            
            # Convertir Z-score a probabilidad de anomalía
            anomaly_score = min(z_score / self.manipulation_thresholds['sentiment_spike'], 1.0)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"⏰ Anomalías temporales {asset}: Score {anomaly_score:.3f}")
            
            return anomaly_score
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis temporal: {e}")
            return 0.0
    
    # Métodos auxiliares de simulación (en implementación real, conectar a APIs reales)
    
    async def _fetch_social_media_data(self, asset: str, timeframe_hours: int) -> List[Dict]:
        """Simula la obtención de datos de redes sociales."""
        # En implementación real, usar Twitter API v2, Reddit API, etc.
        import random
        
        # Simular datos realistas
        num_posts = random.randint(50, 200)
        posts = []
        
        for i in range(num_posts):
            posts.append({
                'id': f"post_{i}",
                'text': f"Análisis de {asset} {'positivo' if random.random() > 0.5 else 'negativo'}",
                'timestamp': datetime.now() - timedelta(hours=random.randint(0, timeframe_hours)),
                'user_id': f"user_{random.randint(1, 100)}",
                'account_age_days': random.randint(1, 1000),
                'followers': random.randint(10, 10000),
                'hashtags': [f"#{asset.lower()}", "#crypto"] if random.random() > 0.3 else []
            })
        
        return posts
    
    async def _fetch_news_timestamps(self, asset: str, timeframe_hours: int) -> List[datetime]:
        """Simula la obtención de timestamps de noticias."""
        import random
        
        num_news = random.randint(3, 10)
        timestamps = []
        
        for _ in range(num_news):
            timestamps.append(datetime.now() - timedelta(hours=random.randint(0, timeframe_hours)))
        
        return sorted(timestamps)
    
    async def _detect_social_spikes(self, asset: str, timeframe_hours: int) -> List[datetime]:
        """Simula la detección de picos de actividad social."""
        import random
        
        num_spikes = random.randint(2, 8)
        spikes = []
        
        for _ in range(num_spikes):
            spikes.append(datetime.now() - timedelta(hours=random.randint(0, timeframe_hours)))
        
        return sorted(spikes)
    
    async def _get_current_activity_level(self, asset: str) -> float:
        """Simula el nivel actual de actividad."""
        import random
        return random.uniform(10, 100)
    
    async def _get_historical_baseline(self, asset: str, days: int) -> List[float]:
        """Simula el baseline histórico de actividad."""
        import random
        return [random.uniform(20, 60) for _ in range(days * 24)]  # Datos por hora
    
    def _calculate_new_account_ratio(self, social_data: List[Dict]) -> float:
        """Calcula la proporción de cuentas nuevas."""
        if not social_data:
            return 0.0
        
        new_accounts = sum(1 for post in social_data 
                          if post['account_age_days'] < self.manipulation_thresholds['account_age_suspicious'])
        
        return new_accounts / len(social_data)
    
    def _calculate_content_similarity(self, social_data: List[Dict]) -> float:
        """Calcula la similitud de contenido entre posts."""
        if len(social_data) < 2:
            return 0.0
        
        # Simplificado: contar palabras comunes
        texts = [post['text'].lower() for post in social_data]
        word_counts = Counter()
        
        for text in texts:
            words = re.findall(r'\w+', text)
            word_counts.update(words)
        
        # Calcular ratio de palabras muy frecuentes
        total_words = sum(word_counts.values())
        common_words = sum(count for word, count in word_counts.most_common(5))
        
        return min(common_words / total_words, 1.0) if total_words > 0 else 0.0
    
    def _detect_hashtag_anomalies(self, social_data: List[Dict], asset: str) -> float:
        """Detecta anomalías en el uso de hashtags."""
        if not social_data:
            return 0.0
        
        hashtag_counts = Counter()
        for post in social_data:
            hashtag_counts.update(post.get('hashtags', []))
        
        asset_hashtag = f"#{asset.lower()}"
        asset_count = hashtag_counts.get(asset_hashtag, 0)
        total_posts = len(social_data)
        
        # Si más del 80% de posts usan el hashtag del activo, es sospechoso
        hashtag_ratio = asset_count / total_posts if total_posts > 0 else 0
        
        return min(hashtag_ratio * 1.25, 1.0) if hashtag_ratio > 0.8 else 0.0
    
    def _analyze_temporal_clustering(self, social_data: List[Dict]) -> float:
        """Analiza clustering temporal de posts."""
        if len(social_data) < 5:
            return 0.0
        
        timestamps = sorted([post['timestamp'] for post in social_data])
        
        # Calcular intervalos entre posts
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutos
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # Si hay muchos posts en intervalos muy cortos, es sospechoso
        short_intervals = sum(1 for interval in intervals if interval < 5)  # menos de 5 minutos
        
        return min(short_intervals / len(intervals) * 2, 1.0)
    
    def _calculate_timing_clustering(self, post_times: List[datetime]) -> float:
        """Calcula clustering de timing entre posts."""
        if len(post_times) < 3:
            return 0.0
        
        # Agrupar por horas
        hour_counts = Counter(dt.hour for dt in post_times)
        
        # Si más del 50% de actividad se concentra en 2-3 horas, es sospechoso
        total_posts = len(post_times)
        top_hours_count = sum(count for hour, count in hour_counts.most_common(3))
        
        concentration_ratio = top_hours_count / total_posts
        
        return min(concentration_ratio * 1.5, 1.0) if concentration_ratio > 0.5 else 0.0
    
    def _analyze_language_patterns(self, social_data: List[Dict]) -> float:
        """Analiza patrones de lenguaje similar."""
        if len(social_data) < 3:
            return 0.0
        
        texts = [post['text'] for post in social_data]
        
        # Contar frases similares (simplificado)
        phrase_counts = Counter()
        for text in texts:
            # Extraer frases de 3+ palabras
            words = text.split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3]).lower()
                phrase_counts[phrase] += 1
        
        # Si hay frases muy repetidas, es sospechoso
        repeated_phrases = sum(1 for phrase, count in phrase_counts.items() if count > 1)
        total_possible_phrases = sum(max(0, len(text.split()) - 2) for text in texts)
        
        return min(repeated_phrases / max(total_possible_phrases, 1) * 5, 1.0)
    
    def _analyze_account_network(self, social_data: List[Dict]) -> float:
        """Analiza la red de cuentas para detectar patrones sospechosos."""
        if len(social_data) < 5:
            return 0.0
        
        # Agrupar por rangos de followers (proxy para detectar cuentas bot)
        follower_ranges = Counter()
        for post in social_data:
            followers = post['followers']
            if followers < 100:
                follower_ranges['low'] += 1
            elif followers < 1000:
                follower_ranges['medium'] += 1
            else:
                follower_ranges['high'] += 1
        
        total_accounts = len(social_data)
        low_follower_ratio = follower_ranges['low'] / total_accounts
        
        # Si más del 70% son cuentas con pocos followers, es sospechoso
        return min(low_follower_ratio * 1.4, 1.0) if low_follower_ratio > 0.7 else 0.0
    
    def _extract_base_asset(self, asset_pair: str) -> str:
        """
        Extrae el activo base de un par de trading.
        
        Args:
            asset_pair: Par completo (ej: 'BTC-USDT', 'OANDA:EUR_USD')
            
        Returns:
            str: Activo base (ej: 'BTC', 'EUR')
        """
        if ':' in asset_pair:
            # Forex pair like 'OANDA:EUR_USD'
            pair = asset_pair.split(':')[1]
            return pair.split('_')[0]
        else:
            # Crypto pair like 'BTC-USDT'
            return asset_pair.split('-')[0]


# Función de prueba
async def test_counter_intel():
    """
    Función de prueba para el motor de contrainteligencia.
    """
    print("🕵️ Probando Counter Intelligence Engine...")
    
    engine = CounterIntelEngine()
    
    # Probar análisis para diferentes activos
    test_assets = ['BTC', 'ETH', 'DOGE']
    
    for asset in test_assets:
        manipulation_score = await engine.analyze_market_manipulation(asset, timeframe_hours=24)
        
        status = "🚨 MANIPULACIÓN DETECTADA" if manipulation_score > 0.7 else \
                "⚠️ ACTIVIDAD SOSPECHOSA" if manipulation_score > 0.5 else \
                "✅ ACTIVIDAD NORMAL"
        
        print(f"{status} - {asset}: Score {manipulation_score:.3f}")
    
    print("🕵️ Prueba del Counter Intelligence Engine completada.")


if __name__ == "__main__":
    asyncio.run(test_counter_intel())
