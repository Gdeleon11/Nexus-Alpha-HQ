
#!/usr/bin/env python3
"""
Reinforcement Learning with Human Feedback (RLHF) Engine
Este módulo analiza los patrones de override del operador para crear un perfil
de confianza dinámico que mejora las decisiones del sistema.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import json
from logger import log_to_dashboard


class RLHFEngine:
    """
    Motor de Aprendizaje por Refuerzo con Feedback Humano.
    Analiza patrones de override para crear perfiles de confianza del operador.
    """
    
    def __init__(self, socketio=None):
        """
        Inicializa el motor RLHF.
        
        Args:
            socketio: Instancia de SocketIO para logs en tiempo real
        """
        self.socketio = socketio
        self.confidence_profile = {}
        self.pattern_weights = {
            'timing_sensitivity': 0.0,
            'spread_sensitivity': 0.0,
            'volatility_sensitivity': 0.0,
            'instinct_reliance': 0.0,
            'market_condition_awareness': 0.0,
            'liquidity_preference': 0.0
        }
        self.asset_preferences = {}
        self.timeframe_preferences = {}
        self.rsi_threshold_adjustments = {}
        self.confidence_multipliers = {}
        
    def analyze_override_patterns(self):
        """
        Analiza los patrones de override del operador desde la base de datos.
        
        Returns:
            dict: Perfil de confianza actualizado del operador
        """
        try:
            if self.socketio:
                log_to_dashboard(self.socketio, "🧠 Iniciando análisis RLHF de patrones de override...")
            
            # Obtener datos de overrides
            override_data = self._get_override_data()
            
            if len(override_data) < 3:  # Necesitamos al menos 3 overrides para análisis
                if self.socketio:
                    log_to_dashboard(self.socketio, f"⚠️ RLHF: Solo {len(override_data)} overrides disponibles. Necesitamos al menos 3 para análisis significativo.")
                return self._get_default_profile()
            
            # Analizar patrones por categoría
            self._analyze_timing_patterns(override_data)
            self._analyze_spread_patterns(override_data)
            self._analyze_volatility_patterns(override_data)
            self._analyze_instinct_patterns(override_data)
            self._analyze_market_condition_patterns(override_data)
            self._analyze_liquidity_patterns(override_data)
            
            # Analizar preferencias por activo y timeframe
            self._analyze_asset_preferences(override_data)
            self._analyze_timeframe_preferences(override_data)
            
            # Analizar ajustes de umbral RSI
            self._analyze_rsi_threshold_adjustments(override_data)
            
            # Calcular multiplicadores de confianza
            self._calculate_confidence_multipliers(override_data)
            
            # Construir perfil final
            profile = self._build_confidence_profile(override_data)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"✅ RLHF: Análisis completado. {len(override_data)} overrides analizados.")
                log_to_dashboard(self.socketio, f"🎯 Perfil de confianza actualizado: {profile['overall_confidence']:.2f}")
            
            return profile
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis RLHF: {e}")
            return self._get_default_profile()
    
    def _get_override_data(self):
        """
        Obtiene datos de overrides desde la base de datos.
        
        Returns:
            list: Lista de registros de override
        """
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        
        # Obtener overrides de los últimos 30 días
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        cursor.execute('''
            SELECT * FROM overrides 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (thirty_days_ago,))
        
        columns = [description[0] for description in cursor.description]
        override_data = []
        
        for row in cursor.fetchall():
            override_record = dict(zip(columns, row))
            override_data.append(override_record)
        
        conn.close()
        return override_data
    
    def _analyze_timing_patterns(self, override_data):
        """Analiza patrones relacionados con timing."""
        timing_overrides = [o for o in override_data if o['reason'] == 'Mal Timing']
        
        if timing_overrides:
            # Analizar en qué timeframes ocurre más 'Mal Timing'
            timeframe_timing_issues = Counter([o['timeframe'] for o in timing_overrides if o['timeframe']])
            total_timing_overrides = len(timing_overrides)
            total_overrides = len(override_data)
            
            # Calcular sensibilidad al timing (0.0 = no sensible, 1.0 = muy sensible)
            self.pattern_weights['timing_sensitivity'] = min(total_timing_overrides / total_overrides, 1.0)
    
    def _analyze_spread_patterns(self, override_data):
        """Analiza patrones relacionados con spread."""
        spread_overrides = [o for o in override_data if o['reason'] == 'Alto Spread']
        
        if spread_overrides:
            total_spread_overrides = len(spread_overrides)
            total_overrides = len(override_data)
            
            self.pattern_weights['spread_sensitivity'] = min(total_spread_overrides / total_overrides, 1.0)
    
    def _analyze_volatility_patterns(self, override_data):
        """Analiza patrones relacionados con volatilidad."""
        volatility_overrides = [o for o in override_data if o['reason'] == 'Volatilidad Inesperada']
        
        if volatility_overrides:
            total_volatility_overrides = len(volatility_overrides)
            total_overrides = len(override_data)
            
            self.pattern_weights['volatility_sensitivity'] = min(total_volatility_overrides / total_overrides, 1.0)
    
    def _analyze_instinct_patterns(self, override_data):
        """Analiza patrones relacionados con instinto del operador."""
        instinct_overrides = [o for o in override_data if o['reason'] == 'Instinto del Operador']
        
        if instinct_overrides:
            total_instinct_overrides = len(instinct_overrides)
            total_overrides = len(override_data)
            
            self.pattern_weights['instinct_reliance'] = min(total_instinct_overrides / total_overrides, 1.0)
    
    def _analyze_market_condition_patterns(self, override_data):
        """Analiza patrones relacionados con condiciones del mercado."""
        market_overrides = [o for o in override_data if o['reason'] == 'Condiciones del Mercado']
        
        if market_overrides:
            total_market_overrides = len(market_overrides)
            total_overrides = len(override_data)
            
            self.pattern_weights['market_condition_awareness'] = min(total_market_overrides / total_overrides, 1.0)
    
    def _analyze_liquidity_patterns(self, override_data):
        """Analiza patrones relacionados con liquidez."""
        liquidity_overrides = [o for o in override_data if o['reason'] == 'Falta de Liquidez']
        
        if liquidity_overrides:
            total_liquidity_overrides = len(liquidity_overrides)
            total_overrides = len(override_data)
            
            self.pattern_weights['liquidity_preference'] = min(total_liquidity_overrides / total_overrides, 1.0)
    
    def _analyze_asset_preferences(self, override_data):
        """Analiza preferencias por tipo de activo."""
        asset_override_counts = Counter([o['asset'] for o in override_data if o['asset']])
        
        # Calcular ratios de override por activo
        for asset, override_count in asset_override_counts.items():
            # Ratio de override (menor ratio = mayor confianza en el activo)
            self.asset_preferences[asset] = {
                'override_count': override_count,
                'confidence_adjustment': max(0.1, 1.0 - (override_count / len(override_data)))
            }
    
    def _analyze_timeframe_preferences(self, override_data):
        """Analiza preferencias por timeframe."""
        timeframe_override_counts = Counter([o['timeframe'] for o in override_data if o['timeframe']])
        
        for timeframe, override_count in timeframe_override_counts.items():
            self.timeframe_preferences[timeframe] = {
                'override_count': override_count,
                'confidence_adjustment': max(0.1, 1.0 - (override_count / len(override_data)))
            }
    
    def _analyze_rsi_threshold_adjustments(self, override_data):
        """Analiza ajustes necesarios en umbrales RSI basados en overrides."""
        rsi_pattern_overrides = []
        
        for override in override_data:
            trigger = override.get('trigger_reason', '')
            if 'RSI' in trigger:
                # Extraer valor RSI del trigger
                rsi_match = re.search(r'RSI.*?(\d+\.?\d*)', trigger)
                if rsi_match:
                    rsi_value = float(rsi_match.group(1))
                    rsi_pattern_overrides.append({
                        'rsi_value': rsi_value,
                        'reason': override['reason'],
                        'asset': override['asset']
                    })
        
        if rsi_pattern_overrides:
            # Analizar patrones de RSI que el operador tiende a ignorar
            ignored_rsi_values = [o['rsi_value'] for o in rsi_pattern_overrides]
            
            if ignored_rsi_values:
                avg_ignored_rsi = sum(ignored_rsi_values) / len(ignored_rsi_values)
                self.rsi_threshold_adjustments = {
                    'avg_ignored_rsi': avg_ignored_rsi,
                    'should_adjust_thresholds': len(ignored_rsi_values) >= 3,
                    'suggested_oversold_threshold': max(25, avg_ignored_rsi - 5) if avg_ignored_rsi < 35 else 30,
                    'suggested_overbought_threshold': min(75, avg_ignored_rsi + 5) if avg_ignored_rsi > 65 else 70
                }
    
    def _calculate_confidence_multipliers(self, override_data):
        """Calcula multiplicadores de confianza basados en patrones."""
        total_overrides = len(override_data)
        
        # Multiplicador basado en frecuencia de overrides
        if total_overrides == 0:
            base_confidence = 1.0
        elif total_overrides <= 5:
            base_confidence = 0.9  # Confianza alta con pocos overrides
        elif total_overrides <= 15:
            base_confidence = 0.8  # Confianza media
        else:
            base_confidence = 0.7  # Confianza baja con muchos overrides
        
        # Ajustar por tipo de override más común
        reason_counts = Counter([o['reason'] for o in override_data])
        most_common_reason = reason_counts.most_common(1)[0] if reason_counts else None
        
        reason_adjustments = {
            'Mal Timing': -0.1,  # Reducir confianza en timing
            'Alto Spread': -0.05,  # Pequeña reducción por spread
            'Volatilidad Inesperada': -0.15,  # Mayor reducción por volatilidad
            'Instinto del Operador': -0.2,  # Mayor reducción por instinto
            'Condiciones del Mercado': -0.1,  # Reducción media por condiciones
            'Falta de Liquidez': -0.05  # Pequeña reducción por liquidez
        }
        
        if most_common_reason:
            reason, count = most_common_reason
            adjustment = reason_adjustments.get(reason, 0.0)
            base_confidence += adjustment * (count / total_overrides)
        
        self.confidence_multipliers = {
            'base_confidence': max(0.1, min(1.0, base_confidence)),
            'total_overrides': total_overrides,
            'most_common_reason': most_common_reason[0] if most_common_reason else None
        }
    
    def _build_confidence_profile(self, override_data):
        """Construye el perfil final de confianza."""
        total_overrides = len(override_data)
        
        # Calcular confianza general
        overall_confidence = self.confidence_multipliers.get('base_confidence', 0.8)
        
        # Crear resumen de patrones
        dominant_pattern = max(self.pattern_weights.items(), key=lambda x: x[1]) if self.pattern_weights else ('ninguno', 0.0)
        
        # Crear perfil estructurado
        profile = {
            'overall_confidence': overall_confidence,
            'total_overrides_analyzed': total_overrides,
            'analysis_timestamp': datetime.now().isoformat(),
            'dominant_override_pattern': {
                'pattern': dominant_pattern[0],
                'strength': dominant_pattern[1]
            },
            'pattern_weights': self.pattern_weights.copy(),
            'asset_preferences': self.asset_preferences.copy(),
            'timeframe_preferences': self.timeframe_preferences.copy(),
            'rsi_adjustments': self.rsi_threshold_adjustments.copy(),
            'recommendations': self._generate_recommendations(),
            'confidence_factors': self._get_confidence_factors()
        }
        
        self.confidence_profile = profile
        return profile
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        # Recomendaciones basadas en patrones dominantes
        if self.pattern_weights.get('timing_sensitivity', 0) > 0.3:
            recommendations.append("Considerar timeframes más largos para reducir sensibilidad al timing")
        
        if self.pattern_weights.get('spread_sensitivity', 0) > 0.2:
            recommendations.append("Implementar filtros de spread antes de ejecutar señales")
        
        if self.pattern_weights.get('volatility_sensitivity', 0) > 0.25:
            recommendations.append("Integrar indicadores de volatilidad adicionales")
        
        if self.pattern_weights.get('instinct_reliance', 0) > 0.4:
            recommendations.append("Refinar algoritmos - el operador confía más en su instinto que en las señales")
        
        # Recomendaciones de ajuste RSI
        if self.rsi_threshold_adjustments.get('should_adjust_thresholds', False):
            avg_rsi = self.rsi_threshold_adjustments['avg_ignored_rsi']
            recommendations.append(f"Ajustar umbrales RSI - valor promedio ignorado: {avg_rsi:.1f}")
        
        return recommendations
    
    def _get_confidence_factors(self):
        """Obtiene factores que afectan la confianza."""
        return {
            'timing_reliability': 1.0 - self.pattern_weights.get('timing_sensitivity', 0),
            'spread_tolerance': 1.0 - self.pattern_weights.get('spread_sensitivity', 0),
            'volatility_handling': 1.0 - self.pattern_weights.get('volatility_sensitivity', 0),
            'algorithmic_trust': 1.0 - self.pattern_weights.get('instinct_reliance', 0),
            'market_awareness': 1.0 - self.pattern_weights.get('market_condition_awareness', 0),
            'liquidity_comfort': 1.0 - self.pattern_weights.get('liquidity_preference', 0)
        }
    
    def _get_default_profile(self):
        """Retorna perfil por defecto cuando no hay suficientes datos."""
        return {
            'overall_confidence': 0.8,  # Confianza neutral por defecto
            'total_overrides_analyzed': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'dominant_override_pattern': {
                'pattern': 'insufficient_data',
                'strength': 0.0
            },
            'pattern_weights': {
                'timing_sensitivity': 0.0,
                'spread_sensitivity': 0.0,
                'volatility_sensitivity': 0.0,
                'instinct_reliance': 0.0,
                'market_condition_awareness': 0.0,
                'liquidity_preference': 0.0
            },
            'asset_preferences': {},
            'timeframe_preferences': {},
            'rsi_adjustments': {},
            'recommendations': ['Recopilar más datos de override para análisis significativo'],
            'confidence_factors': {
                'timing_reliability': 0.8,
                'spread_tolerance': 0.8,
                'volatility_handling': 0.8,
                'algorithmic_trust': 0.8,
                'market_awareness': 0.8,
                'liquidity_comfort': 0.8
            }
        }
    
    def get_signal_confidence_adjustment(self, signal_data):
        """
        Calcula ajuste de confianza para una señal específica.
        
        Args:
            signal_data: Datos de la señal a evaluar
            
        Returns:
            float: Factor de ajuste de confianza (0.1 a 1.0)
        """
        if not self.confidence_profile:
            return 0.8  # Confianza neutral por defecto
        
        adjustment = self.confidence_profile['overall_confidence']
        
        # Ajustar por activo
        asset = signal_data.get('pair', '')
        if asset in self.asset_preferences:
            asset_adjustment = self.asset_preferences[asset]['confidence_adjustment']
            adjustment *= asset_adjustment
        
        # Ajustar por timeframe
        timeframe = signal_data.get('timeframe', '')
        if timeframe in self.timeframe_preferences:
            timeframe_adjustment = self.timeframe_preferences[timeframe]['confidence_adjustment']
            adjustment *= timeframe_adjustment
        
        # Ajustar por RSI si aplica
        trigger = signal_data.get('reason', '')
        if 'RSI' in trigger and self.rsi_threshold_adjustments:
            rsi_match = re.search(r'RSI.*?(\d+\.?\d*)', trigger)
            if rsi_match:
                rsi_value = float(rsi_match.group(1))
                avg_ignored = self.rsi_threshold_adjustments.get('avg_ignored_rsi', 50)
                
                # Si el RSI está cerca del valor promedio ignorado, reducir confianza
                if abs(rsi_value - avg_ignored) < 5:
                    adjustment *= 0.7
        
        return max(0.1, min(1.0, adjustment))
    
    def generate_rlhf_insight(self, signal_data):
        """
        Genera insight RLHF para incluir en el prompt de OpenAI.
        
        Args:
            signal_data: Datos de la señal
            
        Returns:
            str: Insight textual para el prompt
        """
        if not self.confidence_profile:
            return "Perfil RLHF: Datos insuficientes para análisis de patrones del operador."
        
        confidence_adjustment = self.get_signal_confidence_adjustment(signal_data)
        dominant_pattern = self.confidence_profile['dominant_override_pattern']
        
        # Construir mensaje de insight
        insight_parts = []
        
        insight_parts.append(f"Perfil RLHF del Operador (Confianza: {confidence_adjustment:.2f}):")
        
        if dominant_pattern['strength'] > 0.2:
            pattern_name = dominant_pattern['pattern'].replace('_', ' ').title()
            insight_parts.append(f"- Patrón dominante: {pattern_name} ({dominant_pattern['strength']:.2f})")
        
        # Agregar factores específicos
        confidence_factors = self.confidence_profile.get('confidence_factors', {})
        low_confidence_factors = [k.replace('_', ' ').title() for k, v in confidence_factors.items() if v < 0.7]
        
        if low_confidence_factors:
            insight_parts.append(f"- Áreas de menor confianza: {', '.join(low_confidence_factors)}")
        
        # Agregar recomendaciones principales
        recommendations = self.confidence_profile.get('recommendations', [])
        if recommendations:
            main_recommendation = recommendations[0]
            insight_parts.append(f"- Recomendación clave: {main_recommendation}")
        
        return " ".join(insight_parts)


# Función de prueba
async def test_rlhf_engine():
    """Función de prueba para el motor RLHF."""
    print("🧠 Probando RLHF Engine...")
    
    engine = RLHFEngine()
    profile = engine.analyze_override_patterns()
    
    print(f"✅ Perfil generado: {profile['overall_confidence']:.2f} confianza")
    print(f"📊 Overrides analizados: {profile['total_overrides_analyzed']}")
    
    # Probar ajuste de confianza para señal ejemplo
    test_signal = {
        'pair': 'BTC-USDT',
        'timeframe': '5m',
        'reason': 'RSI en sobreventa (28.5)'
    }
    
    adjustment = engine.get_signal_confidence_adjustment(test_signal)
    print(f"🎯 Ajuste de confianza para BTC-USDT: {adjustment:.2f}")
    
    insight = engine.generate_rlhf_insight(test_signal)
    print(f"💡 Insight RLHF: {insight}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rlhf_engine())
