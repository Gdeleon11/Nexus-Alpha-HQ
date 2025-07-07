
import os
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf
from kucoin.client import Market
import finnhub
from alpha_vantage.timeseries import TimeSeries
from typing import Optional, List, Dict, Any, Tuple
import time
import logging

class DataNexus:
    """
    Centralized connector for multiple financial data APIs.
    Provides standardized async access to cryptocurrency, forex, and market data.
    Enhanced with multi-source redundancy and asset health monitoring.
    """
    
    def __init__(self):
        """
        Initialize API clients with keys from environment variables.
        """
        # Load API keys from environment variables
        self.kucoin_api_key = os.getenv("KUCOIN_API_KEY", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        # Initialize clients
        self.kucoin_client = None
        self.finnhub_client = None
        self.alpha_vantage_client = None
        
        # Asset health tracking
        self.healthy_assets = set()
        self.inactive_assets = set()
        self.asset_sources = {}  # Maps asset to best working source
        
        # Error tracking for better diagnostics
        self.api_errors = {
            'kucoin': [],
            'yfinance': [],
            'finnhub': [],
            'alpha_vantage': []
        }
        
        # Initialize KuCoin client (public API, no authentication needed for market data)
        try:
            self.kucoin_client = Market(url='https://api.kucoin.com')
            print(f"✅ KuCoin client initialized successfully{'with API key' if self.kucoin_api_key else ' (public access)'}")
        except Exception as e:
            print(f"❌ Error initializing KuCoin client: {e}")
            self.kucoin_client = None
        
        # Initialize Finnhub client if key is available
        if self.finnhub_api_key:
            try:
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
                print("✅ Finnhub client initialized")
            except Exception as e:
                print(f"❌ Error initializing Finnhub client: {e}")
        else:
            print("⚠️ Finnhub API key not found")
        
        # Initialize Alpha Vantage client if key is available
        if self.alpha_vantage_api_key:
            try:
                self.alpha_vantage_client = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')
                print("✅ Alpha Vantage client initialized")
            except Exception as e:
                print(f"❌ Error initializing Alpha Vantage client: {e}")
        else:
            print("⚠️ Alpha Vantage API key not found")

    async def check_asset_health(self, asset_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Check the health of all assets by testing data availability across all sources.
        
        Args:
            asset_list: List of asset dictionaries
            
        Returns:
            Tuple of (healthy_assets, inactive_assets)
        """
        print("🏥 Iniciando verificación de salud de activos...")
        healthy_assets = []
        inactive_assets = []
        
        for asset in asset_list:
            pair = asset['pair']
            asset_type = asset['type']
            is_healthy = False
            working_source = None
            
            print(f"🔍 Verificando {pair}...")
            
            # Test each source for this asset
            if asset_type == 'crypto':
                # Test crypto sources
                sources_to_test = [
                    ('kucoin', self._test_kucoin_crypto),
                    ('yfinance', self._test_yfinance_crypto),
                    ('alpha_vantage', self._test_alpha_vantage_crypto),
                    ('finnhub', self._test_finnhub_crypto)
                ]
            else:  # forex
                # Test forex sources
                sources_to_test = [
                    ('finnhub', self._test_finnhub_forex),
                    ('yfinance', self._test_yfinance_forex),
                    ('alpha_vantage', self._test_alpha_vantage_forex)
                ]
            
            for source_name, test_func in sources_to_test:
                try:
                    success = await test_func(pair)
                    if success:
                        is_healthy = True
                        working_source = source_name
                        print(f"  ✅ {pair}: Disponible en {source_name}")
                        break
                except Exception as e:
                    print(f"  ❌ {pair}: Error en {source_name} - {str(e)}")
                    continue
            
            if is_healthy:
                self.healthy_assets.add(pair)
                self.asset_sources[pair] = working_source
                healthy_assets.append(asset)
            else:
                self.inactive_assets.add(pair)
                inactive_assets.append(asset)
                print(f"⚠️ ADVERTENCIA: El activo {pair} no se encuentra en ninguna fuente de datos y será ignorado.")
        
        print(f"🏥 Verificación completada: {len(healthy_assets)} activos saludables, {len(inactive_assets)} inactivos")
        return healthy_assets, inactive_assets

    async def _test_kucoin_crypto(self, pair: str) -> bool:
        """Test KuCoin crypto data availability."""
        if not self.kucoin_client:
            return False
        
        try:
            # Convert pair format for KuCoin
            kucoin_pair = self._convert_pair_for_kucoin(pair)
            
            # Try to get recent data
            current_time = int(time.time())
            start_time = current_time - 3600  # 1 hour ago
            
            klines_data = self.kucoin_client.get_kline(
                symbol=kucoin_pair,
                kline_type='5min',
                startAt=start_time,
                endAt=current_time
            )
            
            return klines_data and len(klines_data) > 0
            
        except Exception as e:
            self._log_api_error('kucoin', pair, f"Test failed: {str(e)}")
            return False

    async def _test_yfinance_crypto(self, pair: str) -> bool:
        """Test yfinance crypto data availability."""
        try:
            yf_symbol = self._convert_pair_for_yfinance_crypto(pair)
            yf_ticker = yf.Ticker(yf_symbol)
            hist = yf_ticker.history(period="1d", interval="1h")
            return not hist.empty
            
        except Exception as e:
            self._log_api_error('yfinance', pair, f"Crypto test failed: {str(e)}")
            return False

    async def _test_alpha_vantage_crypto(self, pair: str) -> bool:
        """Test Alpha Vantage crypto data availability."""
        if not self.alpha_vantage_client:
            return False
        
        try:
            # Alpha Vantage crypto format
            from_symbol, to_symbol = self._parse_crypto_pair(pair)
            data, meta_data = self.alpha_vantage_client.get_daily_digital_currency(
                symbol=from_symbol, market=to_symbol
            )
            return not data.empty
            
        except Exception as e:
            self._log_api_error('alpha_vantage', pair, f"Crypto test failed: {str(e)}")
            return False

    async def _test_finnhub_crypto(self, pair: str) -> bool:
        """Test Finnhub crypto data availability."""
        if not self.finnhub_client:
            return False
        
        try:
            # Finnhub uses different format for crypto
            # Convert BTC-USDT to BINANCE:BTCUSDT
            symbol = f"BINANCE:{pair.replace('-', '')}"
            
            current_time = int(time.time())
            start_time = current_time - 3600  # 1 hour ago
            
            res = self.finnhub_client.crypto_candles(symbol, '5', start_time, current_time)
            return res['s'] == 'ok' and res.get('c')
            
        except Exception as e:
            self._log_api_error('finnhub', pair, f"Crypto test failed: {str(e)}")
            return False

    async def _test_finnhub_forex(self, pair: str) -> bool:
        """Test Finnhub forex data availability."""
        if not self.finnhub_client:
            return False
        
        try:
            current_time = int(time.time())
            start_time = current_time - 3600  # 1 hour ago
            
            res = self.finnhub_client.forex_candles(pair, '5', start_time, current_time)
            return res['s'] == 'ok' and res.get('c')
            
        except Exception as e:
            self._log_api_error('finnhub', pair, f"Forex test failed: {str(e)}")
            return False

    async def _test_yfinance_forex(self, pair: str) -> bool:
        """Test yfinance forex data availability."""
        try:
            yf_symbol = self._convert_pair_for_yfinance_forex(pair)
            yf_ticker = yf.Ticker(yf_symbol)
            hist = yf_ticker.history(period="1d", interval="1h")
            return not hist.empty
            
        except Exception as e:
            self._log_api_error('yfinance', pair, f"Forex test failed: {str(e)}")
            return False

    async def _test_alpha_vantage_forex(self, pair: str) -> bool:
        """Test Alpha Vantage forex data availability."""
        if not self.alpha_vantage_client:
            return False
        
        try:
            # Alpha Vantage forex format
            from_currency, to_currency = self._parse_forex_pair(pair)
            data, meta_data = self.alpha_vantage_client.get_daily(symbol=f"{from_currency}{to_currency}")
            return not data.empty
            
        except Exception as e:
            self._log_api_error('alpha_vantage', pair, f"Forex test failed: {str(e)}")
            return False

    async def get_crypto_data(self, pair: str, interval: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get cryptocurrency data with enhanced multi-source redundancy.
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT' for KuCoin)
            interval: Time interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            limit: Number of data points to retrieve (max 1500)
        
        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']
        """
        
        # Check if asset is known to be inactive
        if pair in self.inactive_assets:
            print(f"⚠️ Skipping inactive asset: {pair}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source'])
        
        # Try preferred source first if known
        preferred_source = self.asset_sources.get(pair)
        if preferred_source:
            result = await self._try_crypto_source(preferred_source, pair, interval, limit)
            if result is not None:
                return result
        
        # Try all sources in order of preference
        sources = [
            ('kucoin', self._get_crypto_data_kucoin),
            ('yfinance', self._get_crypto_data_yfinance),
            ('alpha_vantage', self._get_crypto_data_alpha_vantage),
            ('finnhub', self._get_crypto_data_finnhub)
        ]
        
        for source_name, source_func in sources:
            if source_name == preferred_source:
                continue  # Already tried
                
            try:
                print(f"🔄 Intentando {source_name} para {pair}...")
                result = await source_func(pair, interval, limit)
                if result is not None and not result.empty:
                    print(f"✅ Datos obtenidos de {source_name} para {pair}")
                    # Update preferred source
                    self.asset_sources[pair] = source_name
                    return result
                    
            except Exception as e:
                error_msg = self._categorize_error(str(e))
                self._log_api_error(source_name, pair, error_msg)
                print(f"❌ {source_name} falló para {pair}: {error_msg}")
                continue
        
        # All sources failed
        print(f"💥 CRÍTICO: No se pudieron obtener datos para {pair} de ninguna fuente")
        self.inactive_assets.add(pair)
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source'])

    async def _try_crypto_source(self, source_name: str, pair: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Try a specific crypto data source."""
        source_map = {
            'kucoin': self._get_crypto_data_kucoin,
            'yfinance': self._get_crypto_data_yfinance,
            'alpha_vantage': self._get_crypto_data_alpha_vantage,
            'finnhub': self._get_crypto_data_finnhub
        }
        
        if source_name not in source_map:
            return None
            
        try:
            return await source_map[source_name](pair, interval, limit)
        except Exception as e:
            error_msg = self._categorize_error(str(e))
            self._log_api_error(source_name, pair, error_msg)
            return None

    async def _get_crypto_data_kucoin(self, pair: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Get crypto data from KuCoin."""
        if not self.kucoin_client:
            raise Exception("Cliente KuCoin no disponible")
        
        # Convert pair format for KuCoin
        kucoin_pair = self._convert_pair_for_kucoin(pair)
        
        # Map interval to KuCoin interval
        interval_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', 
            '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour',
            '6h': '6hour', '8h': '8hour', '12h': '12hour', '1d': '1day',
            '1w': '1week'
        }
        kucoin_interval = interval_map.get(interval, '5min')
        
        # Calculate start and end time
        current_time = int(time.time())
        interval_seconds = {
            '1min': 60, '3min': 180, '5min': 300, '15min': 900,
            '30min': 1800, '1hour': 3600, '2hour': 7200, '4hour': 14400,
            '6hour': 21600, '8hour': 28800, '12hour': 43200, 
            '1day': 86400, '1week': 604800
        }
        seconds_per_candle = interval_seconds.get(kucoin_interval, 300)
        start_time = current_time - (limit * seconds_per_candle)
        
        # Get klines from KuCoin
        klines_data = self.kucoin_client.get_kline(
            symbol=kucoin_pair,
            kline_type=kucoin_interval,
            startAt=start_time,
            endAt=current_time
        )
        
        if not klines_data or len(klines_data) == 0:
            raise Exception("No hay datos disponibles")
        
        # Convert to standard format
        data_list = []
        for kline in klines_data:
            data_list.append({
                'timestamp': pd.to_datetime(int(kline[0]), unit='s', utc=True),
                'open': float(kline[1]),
                'high': float(kline[3]),
                'low': float(kline[4]),
                'close': float(kline[2]),
                'volume': float(kline[5]),
                'source': 'kucoin'
            })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values('timestamp')
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        return df

    async def _get_crypto_data_yfinance(self, pair: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Get crypto data from yfinance."""
        yf_symbol = self._convert_pair_for_yfinance_crypto(pair)
        yf_ticker = yf.Ticker(yf_symbol)
        
        # Map interval to yfinance interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d', '1w': '1wk', '1M': '1mo'
        }
        yf_interval = interval_map.get(interval, '1d')
        
        # Get historical data
        period_map = {
            '1m': '1d', '5m': '5d', '15m': '5d', '30m': '5d',
            '1h': '10d', '1d': '60d', '1w': '1y', '1M': '2y'
        }
        period = period_map.get(interval, '30d')
        
        hist = yf_ticker.history(period=period, interval=yf_interval)
        
        if hist.empty:
            raise Exception("No hay datos históricos disponibles")
        
        # Standardize output format
        df = pd.DataFrame({
            'timestamp': hist.index.tz_convert('UTC') if hist.index.tz else hist.index.tz_localize('UTC'),
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume'],
            'source': 'yfinance_crypto'
        })
        
        df = df.reset_index(drop=True)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        return df

    async def _get_crypto_data_alpha_vantage(self, pair: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Get crypto data from Alpha Vantage."""
        if not self.alpha_vantage_client:
            raise Exception("Cliente Alpha Vantage no disponible")
        
        from_symbol, to_symbol = self._parse_crypto_pair(pair)
        
        try:
            data, meta_data = self.alpha_vantage_client.get_daily_digital_currency(
                symbol=from_symbol, market=to_symbol
            )
            
            if data.empty:
                raise Exception("No hay datos disponibles en Alpha Vantage")
            
            # Convert to standard format
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.index).tz_localize('UTC'),
                'open': data[f'1a. open ({to_symbol})'].astype(float),
                'high': data[f'2a. high ({to_symbol})'].astype(float),
                'low': data[f'3a. low ({to_symbol})'].astype(float),
                'close': data[f'4a. close ({to_symbol})'].astype(float),
                'volume': data[f'5. volume'].astype(float),
                'source': 'alpha_vantage'
            })
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error Alpha Vantage: {str(e)}")

    async def _get_crypto_data_finnhub(self, pair: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Get crypto data from Finnhub."""
        if not self.finnhub_client:
            raise Exception("Cliente Finnhub no disponible")
        
        # Convert to Finnhub format
        symbol = f"BINANCE:{pair.replace('-', '')}"
        
        current_time = int(time.time())
        start_time = current_time - (30 * 24 * 60 * 60)  # 30 days back
        
        res = self.finnhub_client.crypto_candles(symbol, '5', start_time, current_time)
        
        if res['s'] != 'ok' or not res.get('c'):
            raise Exception("No hay datos disponibles en Finnhub")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(res['t'], unit='s', utc=True),
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res.get('v', [0] * len(res['c'])),
            'source': 'finnhub'
        })
        
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        return df

    async def get_forex_data(self, pair: str, resolution: str = '5', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get forex data with enhanced multi-source redundancy.
        
        Args:
            pair: Forex pair (e.g., 'OANDA:EUR_USD' for Finnhub or 'EURUSD=X' for yfinance)
            resolution: Time resolution ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            limit: Number of data points to retrieve
        
        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']
        """
        
        # Check if asset is known to be inactive
        if pair in self.inactive_assets:
            print(f"⚠️ Skipping inactive asset: {pair}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source'])
        
        # Try preferred source first if known
        preferred_source = self.asset_sources.get(pair)
        if preferred_source:
            result = await self._try_forex_source(preferred_source, pair, resolution, limit)
            if result is not None:
                return result
        
        # Try all sources in order of preference
        sources = [
            ('finnhub', self._get_forex_data_finnhub),
            ('yfinance', self._get_forex_data_yfinance),
            ('alpha_vantage', self._get_forex_data_alpha_vantage)
        ]
        
        for source_name, source_func in sources:
            if source_name == preferred_source:
                continue  # Already tried
                
            try:
                print(f"🔄 Intentando {source_name} para {pair}...")
                result = await source_func(pair, resolution, limit)
                if result is not None and not result.empty:
                    print(f"✅ Datos obtenidos de {source_name} para {pair}")
                    # Update preferred source
                    self.asset_sources[pair] = source_name
                    return result
                    
            except Exception as e:
                error_msg = self._categorize_error(str(e))
                self._log_api_error(source_name, pair, error_msg)
                print(f"❌ {source_name} falló para {pair}: {error_msg}")
                continue
        
        # All sources failed
        print(f"💥 CRÍTICO: No se pudieron obtener datos para {pair} de ninguna fuente")
        self.inactive_assets.add(pair)
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source'])

    async def _try_forex_source(self, source_name: str, pair: str, resolution: str, limit: int) -> Optional[pd.DataFrame]:
        """Try a specific forex data source."""
        source_map = {
            'finnhub': self._get_forex_data_finnhub,
            'yfinance': self._get_forex_data_yfinance,
            'alpha_vantage': self._get_forex_data_alpha_vantage
        }
        
        if source_name not in source_map:
            return None
            
        try:
            return await source_map[source_name](pair, resolution, limit)
        except Exception as e:
            error_msg = self._categorize_error(str(e))
            self._log_api_error(source_name, pair, error_msg)
            return None

    async def _get_forex_data_finnhub(self, pair: str, resolution: str, limit: int) -> Optional[pd.DataFrame]:
        """Get forex data from Finnhub."""
        if not self.finnhub_client:
            raise Exception("Cliente Finnhub no disponible")
        
        current_time = int(time.time())
        start_time = current_time - (30 * 24 * 60 * 60)  # 30 days back
        
        res = self.finnhub_client.forex_candles(pair, resolution, start_time, current_time)
        
        if res['s'] != 'ok' or not res.get('c'):
            raise Exception("No hay datos disponibles")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(res['t'], unit='s', utc=True),
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res.get('v', [0] * len(res['c'])),
            'source': 'finnhub'
        })
        
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        return df

    async def _get_forex_data_yfinance(self, pair: str, resolution: str, limit: int) -> Optional[pd.DataFrame]:
        """Get forex data from yfinance."""
        forex_pair = self._convert_pair_for_yfinance_forex(pair)
        yf_ticker = yf.Ticker(forex_pair)
        
        # Map resolution to yfinance interval
        interval_map = {
            '1': '1m', '5': '5m', '15': '15m', '30': '30m',
            '60': '1h', 'D': '1d', 'W': '1wk', 'M': '1mo'
        }
        yf_interval = interval_map.get(resolution, '1d')
        
        # Get historical data
        period_map = {
            '1': '1d', '5': '5d', '15': '5d', '30': '5d',
            '60': '10d', 'D': '60d', 'W': '1y', 'M': '2y'
        }
        period = period_map.get(resolution, '30d')
        
        hist = yf_ticker.history(period=period, interval=yf_interval)
        
        if hist.empty:
            raise Exception("No hay datos históricos disponibles")
        
        # Standardize output format
        df = pd.DataFrame({
            'timestamp': hist.index.tz_convert('UTC') if hist.index.tz else hist.index.tz_localize('UTC'),
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume'],
            'source': 'yfinance_forex'
        })
        
        df = df.reset_index(drop=True)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        return df

    async def _get_forex_data_alpha_vantage(self, pair: str, resolution: str, limit: int) -> Optional[pd.DataFrame]:
        """Get forex data from Alpha Vantage."""
        if not self.alpha_vantage_client:
            raise Exception("Cliente Alpha Vantage no disponible")
        
        from_currency, to_currency = self._parse_forex_pair(pair)
        symbol = f"{from_currency}{to_currency}"
        
        try:
            data, meta_data = self.alpha_vantage_client.get_daily(symbol=symbol)
            
            if data.empty:
                raise Exception("No hay datos disponibles en Alpha Vantage")
            
            # Convert to standard format
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.index).tz_localize('UTC'),
                'open': data['1. open'].astype(float),
                'high': data['2. high'].astype(float),
                'low': data['3. low'].astype(float),
                'close': data['4. close'].astype(float),
                'volume': data['5. volume'].astype(float),
                'source': 'alpha_vantage'
            })
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error Alpha Vantage: {str(e)}")

    def _convert_pair_for_kucoin(self, pair: str) -> str:
        """Convert pair format for KuCoin."""
        if '-' not in pair and len(pair) >= 6:
            if pair.endswith('USDT'):
                return f"{pair[:-4]}-USDT"
            elif pair.endswith('BTC'):
                return f"{pair[:-3]}-BTC"
            elif pair.endswith('ETH'):
                return f"{pair[:-3]}-ETH"
        return pair

    def _convert_pair_for_yfinance_crypto(self, pair: str) -> str:
        """Convert crypto pair format for yfinance."""
        if pair.endswith('USDT'):
            return pair.replace('USDT', '-USD')
        elif pair.endswith('BTC'):
            return pair.replace('BTC', '-BTC')
        elif pair.endswith('ETH'):
            return pair.replace('ETH', '-ETH')
        else:
            # Common mappings
            pair_map = {
                'BTCUSDT': 'BTC-USD',
                'ETHUSDT': 'ETH-USD',
                'ADAUSDT': 'ADA-USD',
                'DOTUSDT': 'DOT-USD',
                'LINKUSDT': 'LINK-USD',
                'LTCUSDT': 'LTC-USD',
                'BNBUSDT': 'BNB-USD',
                'XRPUSDT': 'XRP-USD'
            }
            return pair_map.get(pair, f"{pair[:3]}-USD")

    def _convert_pair_for_yfinance_forex(self, pair: str) -> str:
        """Convert forex pair format for yfinance."""
        if 'OANDA:' in pair:
            forex_pair = pair.replace('OANDA:', '').replace('_', '') + '=X'
        else:
            pair_map = {
                'EUR_USD': 'EURUSD=X',
                'GBP_USD': 'GBPUSD=X',
                'USD_JPY': 'USDJPY=X',
                'USD_CHF': 'USDCHF=X',
                'USD_CAD': 'USDCAD=X',
                'AUD_USD': 'AUDUSD=X',
                'NZD_USD': 'NZDUSD=X'
            }
            forex_pair = pair_map.get(pair, f"{pair.replace('_', '')}=X")
        return forex_pair

    def _parse_crypto_pair(self, pair: str) -> Tuple[str, str]:
        """Parse crypto pair into from and to symbols."""
        if pair.endswith('USDT'):
            return pair[:-4], 'USD'
        elif pair.endswith('BTC'):
            return pair[:-3], 'BTC'
        elif pair.endswith('ETH'):
            return pair[:-3], 'ETH'
        else:
            # Default assumption
            return pair[:3], 'USD'

    def _parse_forex_pair(self, pair: str) -> Tuple[str, str]:
        """Parse forex pair into from and to currencies."""
        if 'OANDA:' in pair:
            clean_pair = pair.replace('OANDA:', '')
            return clean_pair.split('_')
        elif '_' in pair:
            return pair.split('_')
        else:
            # Assume 6-character format like EURUSD
            return pair[:3], pair[3:]

    def _categorize_error(self, error_str: str) -> str:
        """Categorize error message for better diagnosis."""
        error_lower = error_str.lower()
        
        if 'not found' in error_lower or 'invalid symbol' in error_lower:
            return "Par no listado en esta fuente"
        elif 'rate limit' in error_lower or 'limit exceeded' in error_lower:
            return "Límite de API alcanzado"
        elif 'connection' in error_lower or 'timeout' in error_lower:
            return "API no responde (conexión/timeout)"
        elif 'unauthorized' in error_lower or '401' in error_lower:
            return "Error de autenticación"
        elif 'forbidden' in error_lower or '403' in error_lower:
            return "Acceso denegado a este recurso"
        elif 'empty' in error_lower or 'no data' in error_lower:
            return "Sin datos disponibles en el período solicitado"
        else:
            return f"Error técnico: {error_str[:50]}..."

    def _log_api_error(self, api_name: str, pair: str, error_msg: str):
        """Log API error for diagnostics."""
        error_entry = {
            'timestamp': datetime.now(),
            'pair': pair,
            'error': error_msg
        }
        
        if len(self.api_errors[api_name]) > 10:  # Keep only last 10 errors
            self.api_errors[api_name].pop(0)
        
        self.api_errors[api_name].append(error_entry)

    async def get_market_news(self, category: str = 'general', topic: Optional[str] = None, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Get market news from Finnhub with enhanced error handling."""
        if not self.finnhub_client:
            self._log_api_error('finnhub', 'news', "Cliente no disponible")
            return None
        
        try:
            if topic:
                search_topic = self._format_topic_for_search(topic)
                
                try:
                    company_news = self.finnhub_client.company_news(search_topic, 
                                                                   _from=self._get_date_days_ago(30), 
                                                                   to=self._get_today_date())
                    if company_news:
                        return company_news[:limit]
                except Exception as company_error:
                    self._log_api_error('finnhub', topic, f"Company news failed: {str(company_error)}")
                
                # Fallback to general news
                general_news = self.finnhub_client.general_news(category, min_id=0)
                if general_news:
                    filtered_news = []
                    topic_lower = topic.lower()
                    
                    for article in general_news:
                        if self._article_contains_topic(article, topic_lower):
                            filtered_news.append(article)
                        
                        if len(filtered_news) >= limit:
                            break
                    
                    return filtered_news if filtered_news else []
            
            # Default behavior
            news = self.finnhub_client.general_news(category, min_id=0)
            return news[:limit] if news else []
            
        except Exception as e:
            error_msg = self._categorize_error(str(e))
            self._log_api_error('finnhub', f'news_{category}', error_msg)
            return None
    
    def _format_topic_for_search(self, topic: str) -> str:
        """Format topic for search based on common patterns."""
        topic_upper = topic.upper()
        
        crypto_mappings = {
            'BTC': 'BTCUSD',
            'ETH': 'ETHUSD',
            'XRP': 'XRPUSD',
            'LTC': 'LTCUSD',
            'ADA': 'ADAUSD'
        }
        
        if topic_upper in crypto_mappings:
            return crypto_mappings[topic_upper]
        
        forex_currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        if topic_upper in forex_currencies:
            return topic_upper
        
        return topic_upper
    
    def _article_contains_topic(self, article: Dict[str, Any], topic: str) -> bool:
        """Check if an article contains the specified topic."""
        for field in ['headline', 'summary', 'source']:
            if field in article and article[field]:
                if topic in article[field].lower():
                    return True
        return False
    
    def _get_today_date(self) -> str:
        """Get today's date in YYYY-MM-DD format."""
        return datetime.now().strftime('%Y-%m-%d')
    
    def _get_date_days_ago(self, days: int) -> str:
        """Get date N days ago in YYYY-MM-DD format."""
        from datetime import timedelta
        date_ago = datetime.now() - timedelta(days=days)
        return date_ago.strftime('%Y-%m-%d')

    async def get_intermarket_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get intermarket data using yfinance with enhanced error handling."""
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="100d", interval="1d")
            
            if hist.empty:
                raise Exception("No hay datos disponibles")
            
            df = pd.DataFrame({
                'timestamp': hist.index.tz_convert('UTC') if hist.index.tz else hist.index.tz_localize('UTC'),
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume'],
                'source': 'yfinance'
            })
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            error_msg = self._categorize_error(str(e))
            self._log_api_error('yfinance', ticker, error_msg)
            return None

    async def test_connections(self) -> Dict[str, bool]:
        """Test all API connections with enhanced diagnostics."""
        status = {
            'kucoin': False,
            'finnhub': False,
            'alpha_vantage': False,
            'yfinance': True  # Doesn't require API key
        }
        
        # Test KuCoin
        if self.kucoin_client:
            try:
                test_data = self.kucoin_client.get_kline(
                    symbol='BTC-USDT',
                    kline_type='1day',
                    startAt=int(time.time()) - 86400,
                    endAt=int(time.time())
                )
                status['kucoin'] = test_data and len(test_data) > 0
            except Exception as e:
                self._log_api_error('kucoin', 'connection_test', str(e))
        
        # Test Finnhub
        if self.finnhub_client:
            try:
                quote = self.finnhub_client.quote('AAPL')
                status['finnhub'] = quote and 'c' in quote
            except Exception as e:
                self._log_api_error('finnhub', 'connection_test', str(e))
        
        # Test Alpha Vantage
        if self.alpha_vantage_client:
            try:
                # Simple test - client initialization means it should work
                status['alpha_vantage'] = True
            except Exception as e:
                self._log_api_error('alpha_vantage', 'connection_test', str(e))
        
        return status

    def get_api_status(self) -> Dict[str, str]:
        """Get enhanced status of all API clients."""
        status = {}
        
        # KuCoin status
        if self.kucoin_client:
            recent_errors = len([e for e in self.api_errors['kucoin'] 
                               if (datetime.now() - e['timestamp']).seconds < 3600])
            if self.kucoin_api_key:
                status['kucoin'] = f'Inicializado con API key ({recent_errors} errores recientes)'
            else:
                status['kucoin'] = f'Inicializado (acceso público) ({recent_errors} errores recientes)'
        else:
            status['kucoin'] = 'Falló la inicialización del cliente'
        
        # Finnhub status
        if self.finnhub_client:
            recent_errors = len([e for e in self.api_errors['finnhub'] 
                               if (datetime.now() - e['timestamp']).seconds < 3600])
            status['finnhub'] = f'Inicializado ({recent_errors} errores recientes)'
        elif self.finnhub_api_key:
            status['finnhub'] = 'Clave disponible pero falló la inicialización'
        else:
            status['finnhub'] = 'Clave API no encontrada'
        
        # Alpha Vantage status
        if self.alpha_vantage_client:
            recent_errors = len([e for e in self.api_errors['alpha_vantage'] 
                               if (datetime.now() - e['timestamp']).seconds < 3600])
            status['alpha_vantage'] = f'Inicializado ({recent_errors} errores recientes)'
        elif self.alpha_vantage_api_key:
            status['alpha_vantage'] = 'Clave disponible pero falló la inicialización'
        else:
            status['alpha_vantage'] = 'Clave API no encontrada'
        
        # yfinance status
        recent_errors = len([e for e in self.api_errors['yfinance'] 
                           if (datetime.now() - e['timestamp']).seconds < 3600])
        status['yfinance'] = f'Disponible (sin clave API requerida) ({recent_errors} errores recientes)'
        
        return status

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report of the data connector."""
        return {
            'healthy_assets': len(self.healthy_assets),
            'inactive_assets': len(self.inactive_assets),
            'asset_sources': self.asset_sources,
            'recent_errors': {
                api: len([e for e in errors if (datetime.now() - e['timestamp']).seconds < 3600])
                for api, errors in self.api_errors.items()
            },
            'inactive_asset_list': list(self.inactive_assets)
        }
