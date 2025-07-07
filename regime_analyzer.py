
#!/usr/bin/env python3
"""
Regime Analyzer Module for DataNexus - Market Volatility Regime Detection
Este módulo analiza el régimen actual del mercado basado en el VIX para determinar
si es seguro buscar señales de trading o si debe activarse el modo seguro.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


async def get_market_regime():
    """
    Determina el régimen actual del mercado basado en el VIX.
    
    Returns:
        str: "ALTA_VOLATILIDAD" si VIX > 25, "NORMAL" en caso contrario
    """
    try:
        # Obtener datos del VIX usando yfinance
        vix_ticker = yf.Ticker("^VIX")
        
        # Obtener datos de los últimos 2 días para asegurar que tenemos el valor más reciente
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        vix_data = vix_ticker.history(start=start_date, end=end_date)
        
        if vix_data.empty:
            print("⚠️ No se pudieron obtener datos del VIX. Asumiendo régimen NORMAL.")
            return "NORMAL"
        
        # Obtener el valor más reciente del VIX (precio de cierre)
        current_vix = vix_data['Close'].iloc[-1]
        
        print(f"📊 VIX actual: {current_vix:.2f}")
        
        # Aplicar regla de régimen
        if current_vix > 25:
            return "ALTA_VOLATILIDAD"
        else:
            return "NORMAL"
            
    except Exception as e:
        print(f"❌ Error obteniendo datos del VIX: {e}")
        print("⚠️ Asumiendo régimen NORMAL por seguridad.")
        return "NORMAL"


async def get_dxy_trend():
    """
    Analiza la tendencia del Índice del Dólar (DXY) para análisis de correlación.
    
    Returns:
        str: Descripción de la tendencia del DXY
    """
    try:
        # Obtener datos del DXY usando yfinance
        dxy_ticker = yf.Ticker("DX-Y.NYB")
        
        # Obtener datos de los últimos 10 días para calcular tendencia
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        dxy_data = dxy_ticker.history(start=start_date, end=end_date)
        
        if dxy_data.empty or len(dxy_data) < 5:
            return "Correlación: Datos del DXY insuficientes para análisis"
        
        # Calcular tendencia simple comparando primeros y últimos precios
        initial_price = dxy_data['Close'].iloc[0]
        current_price = dxy_data['Close'].iloc[-1]
        price_change_percent = ((current_price - initial_price) / initial_price) * 100
        
        print(f"📈 DXY: {initial_price:.2f} → {current_price:.2f} ({price_change_percent:+.2f}%)")
        
        # Determinar tendencia
        if price_change_percent > 1.5:
            return f"Correlación: Fuerte tendencia alcista en el DXY ({price_change_percent:+.2f}%) - Positivo para el USD"
        elif price_change_percent < -1.5:
            return f"Correlación: Fuerte tendencia bajista en el DXY ({price_change_percent:+.2f}%) - Negativo para el USD"
        else:
            return f"Correlación: DXY en consolidación ({price_change_percent:+.2f}%) - Impacto neutral en USD"
            
    except Exception as e:
        print(f"❌ Error obteniendo datos del DXY: {e}")
        return "Correlación: Error en análisis del DXY - Impacto neutral asumido"


# Función de prueba
async def test_regime_analyzer():
    """
    Función de prueba para el analizador de régimen.
    """
    print("🧪 Probando Regime Analyzer...")
    
    regime = await get_market_regime()
    print(f"📊 Régimen de mercado detectado: {regime}")
    
    dxy_analysis = await get_dxy_trend()
    print(f"💵 Análisis DXY: {dxy_analysis}")
    
    print("✅ Prueba del Regime Analyzer completada.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_regime_analyzer())
