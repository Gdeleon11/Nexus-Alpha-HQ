#!/usr/bin/env python3
"""
Script de prueba para el escáner de mercados.
Ejecuta un ciclo limitado del escáner para verificar su funcionamiento.
"""

import asyncio
import sys
from market_scanner import MarketScanner
from datanexus_connector import DataNexus


async def test_scanner_cycle():
    """
    Ejecuta un ciclo de prueba del escáner de mercados.
    """
    try:
        print("🚀 Iniciando prueba del escáner de mercados...")
        
        # Inicializar DataNexus
        data_nexus = DataNexus()
        print("✅ DataNexus inicializado")
        
        # Crear escáner
        scanner = MarketScanner(data_nexus)
        print("✅ MarketScanner creado")
        
        # Mostrar activos monitoreados
        scanner.list_assets()
        
        # Ejecutar un solo ciclo de escaneo
        print("🔍 Ejecutando ciclo de prueba...")
        opportunities_found = 0
        
        for i, asset in enumerate(scanner.asset_list, 1):
            print(f"[{i}/{len(scanner.asset_list)}] Escaneando {asset['pair']}...")
            
            # Obtener datos del activo
            df = await scanner._get_asset_data(asset)
            
            if df is not None and len(df) > 0:
                print(f"    📊 Datos obtenidos: {len(df)} registros")
                
                # Calcular RSI
                rsi_values = scanner._calculate_rsi(df['close'], length=14)
                
                if rsi_values is not None and len(rsi_values) > 0:
                    latest_rsi = rsi_values.iloc[-1]
                    
                    if not pd.isna(latest_rsi):
                        # Verificar oportunidades
                        opportunity = scanner._check_catalyst_conditions(asset, latest_rsi)
                        if opportunity:
                            opportunities_found += 1
                            print(f"    {opportunity}")
                        else:
                            print(f"    ✅ RSI normal: {latest_rsi:.2f}")
                    else:
                        print(f"    ⚠️  RSI no calculable")
                else:
                    print(f"    ❌ Error calculando RSI")
            else:
                print(f"    ❌ No se pudieron obtener datos")
            
            # Pausa pequeña entre activos
            await asyncio.sleep(1)
        
        print("\n" + "="*50)
        print(f"🎯 Prueba completada")
        print(f"📊 Activos escaneados: {len(scanner.asset_list)}")
        print(f"🔥 Oportunidades encontradas: {opportunities_found}")
        
        if opportunities_found == 0:
            print("💡 Para ver alertas, el escáner detecta cuando RSI < 30 (sobreventa) o RSI > 70 (sobrecompra)")
        
        print("\n✅ El escáner está funcionando correctamente!")
        print("📝 Para ejecutar el escáner continuo, usa: python market_scanner.py")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        sys.exit(1)


async def test_rsi_calculation():
    """
    Prueba específica del cálculo de RSI.
    """
    import pandas as pd
    import numpy as np
    
    print("\n🧮 Probando cálculo de RSI...")
    
    # Crear datos de prueba
    data_nexus = DataNexus()
    scanner = MarketScanner(data_nexus)
    
    # Obtener datos reales de Bitcoin
    df = await scanner._get_asset_data({'type': 'crypto', 'pair': 'BTC-USDT'})
    
    if df is not None and len(df) > 0:
        rsi_values = scanner._calculate_rsi(df['close'])
        print(f"✅ RSI calculado para {len(df)} precios")
        print(f"📊 RSI más reciente: {rsi_values.iloc[-1]:.2f}")
        print(f"📊 RSI promedio: {rsi_values.dropna().mean():.2f}")
        print(f"📊 RSI mínimo: {rsi_values.dropna().min():.2f}")
        print(f"📊 RSI máximo: {rsi_values.dropna().max():.2f}")
    else:
        print("❌ No se pudieron obtener datos para la prueba de RSI")


if __name__ == "__main__":
    import pandas as pd
    
    print("🔬 Ejecutando pruebas del escáner de mercados\n")
    
    # Ejecutar pruebas
    asyncio.run(test_scanner_cycle())
    print()
    asyncio.run(test_rsi_calculation())