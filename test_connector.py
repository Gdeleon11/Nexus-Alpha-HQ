import asyncio
import os
from datanexus_connector import DataNexus

async def run_tests():
    """
    Script para probar la funcionalidad del Módulo 1: DataNexus_Connector.
    """
    print("INICIANDO INSPECCIÓN DEL MÓDULO 1...")

    # Inicializar el conector
    connector = DataNexus()
    print("Conector DataNexus inicializado.")

    # --- Prueba 1: Datos de Criptomonedas de Binance ---
    print("\n[PRUEBA 1] Obteniendo datos de BTC/USDT de Binance...")
    btc_data = await connector.get_crypto_data('BTCUSDT')
    if btc_data is not None and not btc_data.empty:
        print("✅ Éxito. Datos recibidos de Binance.")
        print("Columnas:", btc_data.columns.to_list())
        print("Primeras 2 filas:")
        print(btc_data.head(2))
    else:
        print("❌ FALLO. No se recibieron datos de Binance.")

    # --- Prueba 2: Datos de Forex de Finnhub ---
    print("\n[PRUEBA 2] Obteniendo datos de EUR/USD de Finnhub (OANDA)...")
    eurusd_data = await connector.get_forex_data('OANDA:EUR_USD')
    if eurusd_data is not None and not eurusd_data.empty:
        print("✅ Éxito. Datos recibidos de Finnhub.")
        print("Columnas:", eurusd_data.columns.to_list())
        print("Primeras 2 filas:")
        print(eurusd_data.head(2))
    else:
        print("❌ FALLO. No se recibieron datos de Finnhub.")

    # --- Prueba 3: Noticias de Mercado de Finnhub ---
    print("\n[PRUEBA 3] Obteniendo noticias de mercado de Finnhub...")
    news = await connector.get_market_news()
    if news:
        print(f"✅ Éxito. Se recibieron {len(news)} noticias.")
        print("Ejemplo de noticia:", news[0])
    else:
        print("❌ FALLO. No se recibieron noticias.")

    # --- Prueba 4: Datos Intermercado de Yahoo Finance ---
    print("\n[PRUEBA 4] Obteniendo datos del S&P 500 (^GSPC) de Yahoo Finance...")
    sp500_data = await connector.get_intermarket_data('^GSPC')
    if sp500_data is not None and not sp500_data.empty:
        print("✅ Éxito. Datos recibidos de Yahoo Finance.")
        print("Columnas:", sp500_data.columns.to_list())
        print(sp500_data.tail(2))
    else:
        print("❌ FALLO. No se recibieron datos de Yahoo Finance.")

    print("\nINSPECCIÓN FINALIZADA.")

if __name__ == "__main__":
    # Esta línea ejecuta nuestro script de pruebas asíncrono
    asyncio.run(run_tests())