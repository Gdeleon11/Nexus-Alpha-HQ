
#!/usr/bin/env python3
"""
Causal Engine for DataNexus - Causal Inference Analysis (Client)
Este módulo actúa como cliente del microservicio de análisis causal.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import requests
import os
import warnings
warnings.filterwarnings('ignore')


def estimate_causal_effect(event_data: Dict[str, Any], market_data: pd.DataFrame) -> float:
    """
    Estima el efecto causal usando el microservicio de análisis causal.

    Args:
        event_data: Diccionario con información del evento
        market_data: DataFrame con datos históricos del mercado

    Returns:
        float: Score de causalidad entre 0 y 1
    """
    try:
        print(f"🔬 Iniciando análisis de inferencia causal via microservicio...")

        # Obtener URL del microservicio desde Secrets
        causal_service_url = os.getenv('CAUSAL_SERVICE_URL', 'http://0.0.0.0:5000')

        # Verificar que los datos del mercado no estén vacíos
        if market_data.empty or len(market_data) < 10:
            print("⚠️ Datos insuficientes para análisis causal")
            return 0.1

        # Preparar payload para el microservicio
        # Convertir DataFrame a diccionario para JSON
        market_data_dict = market_data.to_dict('list')

        payload = {
            'event_data': event_data,
            'market_data': market_data_dict
        }

        # Hacer solicitud al microservicio
        response = requests.post(
            f"{causal_service_url}/analyze",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            causal_score = result.get('causal_score', 0.1)
            interpretation = result.get('interpretation', 'Sin interpretación')

            print(f"🔬 Microservicio causal: {interpretation} (Score: {causal_score:.3f})")
            return causal_score
        else:
            print(f"⚠️ Error en microservicio causal: {response.status_code}")
            # Fallback local
            return _local_fallback_analysis(event_data, market_data)

    except requests.exceptions.RequestException as e:
        print(f"🔌 Microservicio causal no disponible: {e}")
        # Fallback local
        return _local_fallback_analysis(event_data, market_data)
    except Exception as e:
        print(f"❌ Error en análisis causal: {e}")
        return 0.1


def _local_fallback_analysis(event_data: Dict[str, Any], market_data: pd.DataFrame) -> float:
    """
    Análisis causal simple como fallback cuando el microservicio no está disponible.

    Args:
        event_data: Información del evento
        market_data: Datos del mercado

    Returns:
        Score causal simplificado
    """
    try:
        if market_data.empty or len(market_data) < 10:
            return 0.1

        # Análisis estadístico básico
        price_changes = market_data['close'].pct_change().dropna()

        # Calcular volatilidad
        volatility = price_changes.std()

        # Estimar impacto basado en intensidad del evento
        event_intensity = event_data.get('intensity', 0.5)
        sentiment_factor = 1.0

        if event_data.get('sentiment') == 'positive':
            sentiment_factor = 1.2
        elif event_data.get('sentiment') == 'negative':
            sentiment_factor = 1.2

        # Score simple basado en volatilidad y características del evento
        base_score = min(1.0, volatility * 10) * event_intensity * sentiment_factor

        score = max(0.1, min(1.0, base_score))
        print(f"📊 Análisis causal local (fallback): Score {score:.3f}")
        return score

    except Exception as e:
        print(f"Error en análisis fallback: {e}")
        return 0.1


class CausalEngine:
    """
    Cliente del microservicio de inferencia causal.
    """

    def __init__(self, service_url: str = None):
        """
        Inicializa el cliente del motor de inferencia causal.

        Args:
            service_url: URL del microservicio (opcional)
        """
        # Obtener URL desde Secrets o usar la proporcionada
        self.service_url = service_url or os.getenv('CAUSAL_SERVICE_URL')

        if not self.service_url:
            print("⚠️ CAUSAL_SERVICE_URL no está configurado en Secrets")
            self.service_available = False
            return

        # Verificar conectividad con el microservicio
        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Conectado al microservicio causal: {health_data.get('available_methods', [])}")
                self.service_available = True
            else:
                print(f"⚠️ Microservicio causal respondió con código: {response.status_code}")
                self.service_available = False
        except Exception as e:
            print(f"⚠️ No se pudo conectar al microservicio causal: {e}")
            self.service_available = False

    def estimate_causal_effect(self, event_data: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """
        Estima el efecto causal usando el microservicio.

        Args:
            event_data: Diccionario con información del evento
            market_data: DataFrame con datos de precios del mercado

        Returns:
            Score causal de 0 a 1
        """
        if not self.service_available:
            print("⚠️ Microservicio no disponible, usando análisis fallback local")
            return _local_fallback_analysis(event_data, market_data)

        return estimate_causal_effect(event_data, market_data)


# Función de prueba
def test_causal_engine():
    """
    Función de prueba para el cliente del motor causal.
    """
    print("🧪 Probando Cliente del Motor de Inferencia Causal...")

    # Crear datos de prueba
    np.random.seed(42)
    n_points = 50

    # Simular datos de mercado
    base_price = 100
    prices = [base_price]
    volumes = []

    for i in range(n_points - 1):
        if i == 25:
            shock = 5
        else:
            shock = 0

        change = np.random.normal(0, 1) + shock * 0.5
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
        volumes.append(np.random.randint(1000, 10000))

    # Crear DataFrame de mercado
    market_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='1H'),
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    })

    # Crear evento de prueba
    event_data = {
        'type': 'news',
        'sentiment': 'positive',
        'timestamp': '2024-01-01T01:00:00Z',
        'intensity': 0.8,
        'description': 'Noticia importante sobre el activo'
    }

    # Ejecutar análisis causal
    causal_score = estimate_causal_effect(event_data, market_data)

    print(f"📊 Score Causal Calculado: {causal_score:.3f}")
    print(f"🎯 Interpretación: {'Alta causalidad' if causal_score > 0.7 else 'Causalidad moderada' if causal_score > 0.4 else 'Baja causalidad'}")
    print("✅ Prueba del Cliente Motor Causal completada!")


if __name__ == "__main__":
    test_causal_engine()
