import asyncio
import time
import pandas as pd
import random
import threading
from queue import Queue, Empty
from datetime import datetime
from datanexus_connector import DataNexus
from deep_analysis_engine import DeepAnalysisEngine
from logger import log_to_dashboard, send_signal_to_dashboard
from regime_analyzer import get_market_regime
# Asumimos que evolutionary_engine.py y su función existen
from evolutionary_engine import evolve_trading_strategy

# Try to import pandas_ta, use manual calculation as fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print(
        "Warning: pandas_ta not available, using manual indicator calculations"
    )


class MarketScanner:

    def __init__(self, data_connector, socketio, priority_queue):
        self.data_connector = data_connector
        self.socketio = socketio  # Guardar la instancia de socketio
        self.priority_queue = priority_queue  # Guardar la instancia de la cola
        self.analysis_engine = DeepAnalysisEngine(self.data_connector, self.socketio)

        # Cola de comandos prioritarios locales (mantenida por compatibilidad)
        self.priority_command_queue = Queue()
        self.command_lock = threading.Lock()

        self.asset_list = [
            # --- Criptomonedas Principales ---
            {'type': 'crypto', 'pair': 'BTC-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'ETH-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'SOL-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'XRP-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'ADA-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'DOGE-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'AVAX-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'LINK-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'DOT-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'MATIC-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'LTC-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'UNI-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'ATOM-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'ALGO-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'VET-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'FTM-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'NEAR-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'MANA-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'SAND-USDT', 'timeframes': ['1m', '5m', '15m']},
            {'type': 'crypto', 'pair': 'SHIB-USDT', 'timeframes': ['1m', '5m', '15m']},

            # --- Forex Principales ---
            {'type': 'forex', 'pair': 'OANDA:EUR_USD', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:GBP_USD', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:USD_JPY', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:USD_CHF', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:AUD_USD', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:USD_CAD', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:NZD_USD', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:EUR_GBP', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:EUR_JPY', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:GBP_JPY', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:AUD_JPY', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:EUR_CHF', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:GBP_CHF', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:CHF_JPY', 'timeframes': ['5m', '15m', '30m']},
            {'type': 'forex', 'pair': 'OANDA:CAD_JPY', 'timeframes': ['5m', '15m', '30m']},

            # --- Forex Exóticos ---
            {'type': 'forex', 'pair': 'OANDA:EUR_AUD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:EUR_CAD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:EUR_NZD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:GBP_AUD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:GBP_CAD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:GBP_NZD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:AUD_CAD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:AUD_CHF', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:AUD_NZD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:CAD_CHF', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:NZD_CAD', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:NZD_CHF', 'timeframes': ['15m', '30m', '1h']},
            {'type': 'forex', 'pair': 'OANDA:NZD_JPY', 'timeframes': ['15m', '30m', '1h']},
        ]

        self.evolved_strategies = {}
        self.strategies_initialized = False

    async def execute_strategic_priority_analysis(self, task_data):
        """
        Ejecuta análisis prioritario solicitado por el Director Estratégico.

        Args:
            task_data: Datos de la tarea prioritaria
        """
        try:
            pair = task_data.get('pair')
            timeframe = task_data.get('timeframe', '5m')

            log_to_dashboard(self.socketio, f"⚡ ANÁLISIS PRIORITARIO DIRECTOR: {pair} ({timeframe})")

            # Buscar el activo en nuestra lista
            target_asset = None
            for asset in self.asset_list:
                if asset['pair'] == pair:
                    target_asset = asset
                    break

            if not target_asset:
                log_to_dashboard(self.socketio, f"❌ Activo no encontrado en lista: {pair}")
                return

            # Crear oportunidad sintética para análisis prioritario
            opportunity = {
                'pair': pair,
                'type': target_asset['type'],
                'timeframe': timeframe,
                'reason': task_data.get('reason', f'{pair}: Análisis prioritario del Director Estratégico'),
                'strategy': 'strategic_priority',
                'priority': True
            }

            log_to_dashboard(self.socketio, f"🧠 Iniciando análisis profundo prioritario para {pair}...")

            # Ejecutar análisis profundo inmediatamente
            await self.analysis_engine.analyze_opportunity(opportunity)

            log_to_dashboard(self.socketio, f"✅ Análisis prioritario completado: {pair}")

        except Exception as e:
            log_to_dashboard(self.socketio, f"❌ Error en análisis prioritario: {str(e)}")

    async def initialize_evolved_strategies(self):
        if self.strategies_initialized:
            return

        log_to_dashboard(
            self.socketio,
            "🏥 Verificando salud de activos antes de evolucionar estrategias...")

        # Check asset health first
        healthy_assets, inactive_assets = await self.data_connector.check_asset_health(self.asset_list)

        if inactive_assets:
            log_to_dashboard(
                self.socketio,
                f"⚠️ {len(inactive_assets)} activos marcados como inactivos y serán ignorados")

        # Update asset list to only include healthy assets
        self.asset_list = healthy_assets

        log_to_dashboard(
            self.socketio,
            f"🧬 Iniciando evolución de estrategias para {len(self.asset_list)} activos saludables...")

        for asset in self.asset_list:
            pair = asset['pair']
            timeframes = asset.get('timeframes', ['5m'])

            for timeframe in timeframes:
                strategy_key = f"{pair}_{timeframe}"
                log_to_dashboard(self.socketio,
                                 f"🔬 Evolucionando estrategia para {pair} ({timeframe})...")

                try:
                    # Intentar obtener indicadores Alpha Genesis primero
                    alpha_strategy = self._get_alpha_genesis_strategy(pair, timeframe)

                    if alpha_strategy:
                        self.evolved_strategies[strategy_key] = alpha_strategy
                        log_to_dashboard(
                            self.socketio,
                            f"✅ Usando indicador Alpha Genesis para {pair} ({timeframe}): {alpha_strategy.get('alpha_id', 'Unknown')}")
                    else:
                        # Estrategia de respaldo tradicional
                        self.evolved_strategies[strategy_key] = {
                            'indicator': 'RSI',
                            'entry_level': 30,
                            'exit_level': 70,
                            'rsi_period': 14,
                            'timeframe': timeframe
                        }
                    log_to_dashboard(
                        self.socketio,
                        f"✅ Usando estrategia de respaldo para {pair} ({timeframe}): RSI")
                except Exception as e:
                    log_to_dashboard(
                        self.socketio,
                        f"❌ Error evolucionando estrategia para {pair} ({timeframe}): {e}")
                    self.evolved_strategies[strategy_key] = {
                        'indicator': 'RSI',
                        'entry_level': 30,
                        'exit_level': 70,
                        'rsi_period': 14,
                        'timeframe': timeframe
                    }

        self.strategies_initialized = True
        log_to_dashboard(self.socketio,
                         "🎯 Evolución de estrategias completada!")

        # Log health report
        health_report = self.data_connector.get_health_report()
        log_to_dashboard(self.socketio,
                         f"📊 Reporte de salud: {health_report['healthy_assets']} activos saludables, {health_report['inactive_assets']} inactivos")

        if health_report['inactive_asset_list']:
            log_to_dashboard(self.socketio,
                             f"🚫 Activos inactivos: {', '.join(health_report['inactive_asset_list'][:5])}{'...' if len(health_report['inactive_asset_list']) > 5 else ''}")

        log_to_dashboard(self.socketio, "=" * 50)

    async def scan_markets(self):
        await self.initialize_evolved_strategies()

        log_to_dashboard(self.socketio, "🔍 Iniciando escáner de mercados...")
        log_to_dashboard(self.socketio,
                         f"📊 Monitoreando {len(self.asset_list)} activos")
        log_to_dashboard(self.socketio, "=" * 50)

        while True:
            # 1. VERIFICAR ÓRDENES PRIORITARIAS
            if not self.priority_queue.empty():
                priority_task = self.priority_queue.get()
                log_to_dashboard(self.socketio, f"🚨 ORDEN PRIORITARIA RECIBIDA. INTERRUMPIENDO ESCANEO.")
                await self.analysis_engine.analyze_opportunity(priority_task)
                log_to_dashboard(self.socketio, "✅ Tarea prioritaria completada. Reanudando ciclo normal.")
                self.priority_queue.task_done()
                continue  # Vuelve al inicio del bucle para buscar más órdenes

            # 2. SI NO HAY ÓRDENES, PROCEDER CON EL ESCANEO NORMAL
            log_to_dashboard(self.socketio, "--- No hay órdenes prioritarias. Iniciando ciclo de escaneo normal...")

            scan_start_time = time.time()
            opportunities_found = 0

            # Verificar régimen de mercado antes de escanear
            market_regime = await get_market_regime()
            log_to_dashboard(self.socketio, f"📊 Régimen de mercado: {market_regime}")

            if market_regime == "ALTA_VOLATILIDAD":
                log_to_dashboard(self.socketio, "⚠️ Modo Seguro activado. Volatilidad muy alta. No se buscarán señales.")
                log_to_dashboard(self.socketio, "⏳ Esperando 60 segundos para el próximo chequeo de régimen...")
                log_to_dashboard(self.socketio, "=" * 50)
                await asyncio.sleep(60)
                continue

            for asset in self.asset_list:
                pair = asset['pair']
                timeframes = asset.get('timeframes', ['5m'])

                for timeframe in timeframes:
                    log_to_dashboard(self.socketio,
                                     f"--- Escaneando {pair} ({timeframe})...")

                    # Obtener datos según el tipo de activo y timeframe
                    if asset['type'] == 'crypto':
                        # Mapear timeframe a intervalo de crypto
                        crypto_interval = timeframe
                        df = await self.data_connector.get_crypto_data(
                            pair, interval=crypto_interval)
                    elif asset['type'] == 'forex':
                        # Mapear timeframe a resolución de forex
                        resolution_map = {
                            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                            '1h': '60', '4h': '240', '1d': 'D'
                        }
                        forex_resolution = resolution_map.get(timeframe, '5')
                        df = await self.data_connector.get_forex_data(
                            pair, resolution=forex_resolution)
                    else:
                        df = None

                    if df is not None and not df.empty:
                        strategy_key = f"{pair}_{timeframe}"
                        strategy = self.evolved_strategies.get(strategy_key)
                        if strategy:
                            # Calcular RSI - usar pandas-ta si está disponible, sino manual
                            if PANDAS_TA_AVAILABLE:
                                df.ta.rsi(length=14, append=True)
                                last_rsi = df['RSI_14'].iloc[-1]
                            else:
                                last_rsi = self._calculate_rsi_manual(
                                    df['close'], 14)

                            if pd.isna(last_rsi):
                                continue

                            # --- MODIFICACIÓN DE PRUEBA ---
                            # Umbral temporalmente sensible para forzar una señal
                            if last_rsi < 30 or last_rsi > 70:
                                opportunity_message = f"{pair}: Condición de prueba RSI ({last_rsi:.2f})"
                                log_to_dashboard(
                                    self.socketio,
                                    f"🚨 OPORTUNIDAD DETECTADA ({timeframe}): {opportunity_message}"
                                )
                                opportunities_found += 1

                                # === INICIO DE BLOQUE DE PRUEBA DEL DASHBOARD ===
                                log_to_dashboard(
                                    self.socketio,
                                    f"--- ENVIANDO MENSAJE DE PRUEBA DIRECTO AL DASHBOARD ({timeframe}) ---"
                                )
                                test_card = {
                                    'pair': pair,
                                    'timeframe': timeframe,
                                    'trigger': opportunity_message,
                                    'cognitive': 'TEST',
                                    'predictive': {
                                        'prediction': 'TEST',
                                        'confidence': 1
                                    },
                                    'macro': 'TEST',
                                    'social': 'TEST',
                                    'verdict': 'PRUEBA DE CONEXIÓN'
                                }
                                send_signal_to_dashboard(self.socketio, test_card)
                                # === FIN DE BLOQUE DE PRUEBA ===

                                opportunity = {
                                    'pair': pair,
                                    'type': asset['type'],
                                    'timeframe': timeframe,
                                    'reason': opportunity_message,
                                    'strategy': strategy.get('indicator', 'Unknown')
                                }
                                await self.analysis_engine.analyze_opportunity(
                                    opportunity)
                            else:
                                log_to_dashboard(
                                    self.socketio,
                                    f"    ✅ {pair} ({timeframe}): RSI normal ({last_rsi:.2f})"
                                )
                        else:
                            log_to_dashboard(
                                self.socketio,
                                f"    ⚠️ {pair} ({timeframe}): No hay estrategia evolucionada"
                            )
                    else:
                        log_to_dashboard(
                            self.socketio,
                            f"    ❌ {pair} ({timeframe}): No se pudieron obtener datos")

                    await asyncio.sleep(0.5)  # Reduced sleep between timeframes

            scan_duration = time.time() - scan_start_time
            log_to_dashboard(self.socketio, "-" * 50)
            log_to_dashboard(self.socketio,
                             f"📈 Escaneo completado en {scan_duration:.2f}s")
            log_to_dashboard(
                self.socketio,
                f"🔥 Oportunidades encontradas: {opportunities_found}")
            log_to_dashboard(
                self.socketio,
                f"⏳ Esperando 60 segundos para el próximo escaneo...")
            log_to_dashboard(self.socketio, "=" * 50)

            await asyncio.sleep(60)

    def _calculate_rsi_manual(self, prices, period=14):
        """
        Calculate RSI manually when pandas_ta is not available.

        Args:
            prices: Series of closing prices
            period: RSI period

        Returns:
            RSI value (float)
        """
        if len(prices) < period + 1:
            return 50  # Default neutral RSI if not enough data

        deltas = prices.diff()[1:]  # Remove first NaN value
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        # Calculate initial averages
        avg_gain = gains.iloc[:period].mean()
        avg_loss = losses.iloc[:period].mean()

        # Calculate smoothed averages
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _get_alpha_genesis_strategy(self, pair, timeframe):
        """
        Obtiene una estrategia basada en indicadores Alpha Genesis.

        Args:
            pair: Par de trading
            timeframe: Timeframe

        Returns:
            Diccionario con estrategia Alpha Genesis o None
        """
        try:
            from alpha_genesis_engine import AlphaGenesisEngine
            alpha_genesis = AlphaGenesisEngine()

            # Obtener mejores indicadores
            best_indicators = alpha_genesis.get_best_indicators(3)

            if best_indicators:
                # Seleccionar indicador aleatorio de los mejores
                selected = random.choice(best_indicators)

                config = alpha_genesis.export_indicator_for_evolution(selected['id'])
                if config:
                    return {
                        'indicator': 'alpha_genesis',
                        'alpha_id': selected['id'],
                        'alpha_expression': config['expression'],
                        'alpha_fitness': config['fitness_score'],
                        'entry_threshold': -1.0,
                        'exit_threshold': 1.0,
                        'timeframe': timeframe
                    }

            return None

        except Exception as e:
            log_to_dashboard(
                self.socketio,
                f"⚠️ Error obteniendo estrategia Alpha Genesis: {str(e)}")
            return None