#!/usr/bin/env python3
"""
Backtester Module for DataNexus - Trading Strategy Performance Simulation
Este módulo permite simular estrategias de trading y calcular métricas de rendimiento.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import pandas_ta, if it fails, use manual calculations
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not available, using manual indicator calculations")

class BacktesterEngine:
    """
    Motor de backtesting que simula estrategias de trading y calcula métricas de rendimiento.
    """
    
    def __init__(self):
        """Inicializa el motor de backtesting."""
        self.trades = []
        self.current_position = None
        self.portfolio_value = []
        self.initial_capital = 10000  # Capital inicial por defecto
        self.current_capital = self.initial_capital
        
    def run_backtest(self, strategy: Dict[str, Any], historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta una simulación de backtesting con la estrategia especificada.
        
        Args:
            strategy: Diccionario con parámetros de la estrategia
                     Ejemplo: {'indicator': 'RSI', 'entry_level': 30, 'exit_level': 50}
            historical_data: DataFrame con datos OHLCV
                           Columnas requeridas: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        print("🚀 Iniciando simulación de backtesting...")
        print(f"📊 Estrategia: {strategy}")
        print(f"📈 Datos históricos: {len(historical_data)} velas")
        
        # Resetear estado
        self._reset_state()
        
        # Validar datos
        if not self._validate_data(historical_data):
            return self._get_empty_results("Error: Datos inválidos")
        
        # Preparar datos con indicadores
        data_with_indicators = self._prepare_indicators(historical_data, strategy)
        
        if data_with_indicators is None:
            return self._get_empty_results("Error: No se pudieron calcular indicadores")
        
        # Ejecutar simulación vela por vela
        for i in range(len(data_with_indicators)):
            current_candle = data_with_indicators.iloc[i]
            self._process_candle(current_candle, strategy, i)
            
            # Registrar valor del portafolio
            portfolio_val = self._calculate_portfolio_value(current_candle)
            self.portfolio_value.append(portfolio_val)
        
        # Cerrar posición abierta si existe
        if self.current_position:
            final_candle = data_with_indicators.iloc[-1]
            self._close_position(final_candle, "Final del backtest")
        
        # Calcular métricas
        results = self._calculate_metrics()
        
        print("✅ Backtesting completado")
        print(f"📊 Operaciones totales: {results['total_trades']}")
        print(f"🎯 Tasa de éxito: {results['win_rate']:.2f}%")
        print(f"💰 Factor de ganancia: {results['profit_factor']:.2f}")
        print(f"📉 Máximo drawdown: {results['max_drawdown']:.2f}%")
        
        return results
    
    def _reset_state(self):
        """Resetea el estado del backtester."""
        self.trades = []
        self.current_position = None
        self.portfolio_value = []
        self.current_capital = self.initial_capital
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Valida que los datos históricos tengan el formato correcto.
        
        Args:
            data: DataFrame a validar
            
        Returns:
            True si los datos son válidos
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            print(f"❌ Error: Columnas requeridas: {required_columns}")
            print(f"📊 Columnas disponibles: {list(data.columns)}")
            return False
        
        if len(data) < 20:
            print("❌ Error: Se necesitan al menos 20 velas para el backtesting")
            return False
        
        return True
    
    def _prepare_indicators(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepara los datos añadiendo indicadores técnicos según la estrategia.
        
        Args:
            data: DataFrame con datos OHLCV
            strategy: Diccionario con parámetros de la estrategia
            
        Returns:
            DataFrame con indicadores añadidos o None si hay error
        """
        try:
            df = data.copy()
            
            # Convertir timestamp a datetime si es string
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Añadir indicadores según la estrategia
            indicator = strategy.get('indicator', 'RSI').upper()
            
            if indicator == 'RSI':
                period = strategy.get('rsi_period', 14)
                if HAS_PANDAS_TA:
                    df['RSI'] = ta.rsi(df['close'], length=period)
                else:
                    df['RSI'] = self._calculate_rsi_manual(df['close'], period)
            elif indicator == 'MACD':
                if HAS_PANDAS_TA:
                    macd_data = ta.macd(df['close'])
                    df['MACD'] = macd_data['MACD_12_26_9']
                    df['MACD_Signal'] = macd_data['MACDs_12_26_9']
                    df['MACD_Histogram'] = macd_data['MACDh_12_26_9']
                else:
                    macd_data = self._calculate_macd_manual(df['close'])
                    df['MACD'] = macd_data['MACD']
                    df['MACD_Signal'] = macd_data['Signal']
                    df['MACD_Histogram'] = macd_data['Histogram']
            elif indicator == 'SMA':
                fast_period = strategy.get('sma_fast', 10)
                slow_period = strategy.get('sma_slow', 20)
                if HAS_PANDAS_TA:
                    df['SMA_Fast'] = ta.sma(df['close'], length=fast_period)
                    df['SMA_Slow'] = ta.sma(df['close'], length=slow_period)
                else:
                    df['SMA_Fast'] = df['close'].rolling(window=fast_period).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_period).mean()
            
            # Eliminar filas con NaN (período de calentamiento de indicadores)
            df = df.dropna().reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error calculando indicadores: {str(e)}")
            return None
    
    def _process_candle(self, candle: pd.Series, strategy: Dict[str, Any], index: int):
        """
        Procesa una vela individual aplicando la lógica de la estrategia.
        
        Args:
            candle: Serie con datos de la vela actual
            strategy: Diccionario con parámetros de la estrategia
            index: Índice de la vela actual
        """
        indicator = strategy.get('indicator', 'RSI').upper()
        
        if indicator == 'RSI':
            self._process_rsi_strategy(candle, strategy, index)
        elif indicator == 'MACD':
            self._process_macd_strategy(candle, strategy, index)
        elif indicator == 'SMA':
            self._process_sma_strategy(candle, strategy, index)
    
    def _process_rsi_strategy(self, candle: pd.Series, strategy: Dict[str, Any], index: int):
        """
        Procesa estrategia basada en RSI.
        
        Args:
            candle: Datos de la vela actual
            strategy: Parámetros de la estrategia
            index: Índice de la vela
        """
        rsi = candle['RSI']
        entry_level = strategy.get('entry_level', 30)
        exit_level = strategy.get('exit_level', 70)
        
        # Lógica de entrada (RSI en sobreventa)
        if not self.current_position and rsi <= entry_level:
            self._open_position(candle, 'BUY', f'RSI sobreventa ({rsi:.2f})')
        
        # Lógica de salida (RSI en sobrecompra o stop loss)
        elif self.current_position and rsi >= exit_level:
            self._close_position(candle, f'RSI sobrecompra ({rsi:.2f})')
    
    def _process_macd_strategy(self, candle: pd.Series, strategy: Dict[str, Any], index: int):
        """
        Procesa estrategia basada en MACD.
        
        Args:
            candle: Datos de la vela actual
            strategy: Parámetros de la estrategia
            index: Índice de la vela
        """
        macd = candle['MACD']
        signal = candle['MACD_Signal']
        
        # Lógica de entrada (MACD cruza por encima de la señal)
        if not self.current_position and macd > signal:
            self._open_position(candle, 'BUY', f'MACD cruce alcista ({macd:.4f} > {signal:.4f})')
        
        # Lógica de salida (MACD cruza por debajo de la señal)
        elif self.current_position and macd < signal:
            self._close_position(candle, f'MACD cruce bajista ({macd:.4f} < {signal:.4f})')
    
    def _process_sma_strategy(self, candle: pd.Series, strategy: Dict[str, Any], index: int):
        """
        Procesa estrategia basada en medias móviles.
        
        Args:
            candle: Datos de la vela actual
            strategy: Parámetros de la estrategia
            index: Índice de la vela
        """
        sma_fast = candle['SMA_Fast']
        sma_slow = candle['SMA_Slow']
        
        # Lógica de entrada (SMA rápida cruza por encima de SMA lenta)
        if not self.current_position and sma_fast > sma_slow:
            self._open_position(candle, 'BUY', f'SMA cruce alcista ({sma_fast:.2f} > {sma_slow:.2f})')
        
        # Lógica de salida (SMA rápida cruza por debajo de SMA lenta)
        elif self.current_position and sma_fast < sma_slow:
            self._close_position(candle, f'SMA cruce bajista ({sma_fast:.2f} < {sma_slow:.2f})')
    
    def _open_position(self, candle: pd.Series, direction: str, reason: str):
        """
        Abre una nueva posición.
        
        Args:
            candle: Datos de la vela actual
            direction: Dirección de la operación ('BUY' o 'SELL')
            reason: Razón de la entrada
        """
        self.current_position = {
            'entry_time': candle['timestamp'],
            'entry_price': candle['close'],
            'direction': direction,
            'reason': reason,
            'quantity': self.current_capital / candle['close']  # Usar todo el capital
        }
        
        print(f"🟢 Entrada {direction}: {candle['close']:.2f} - {reason}")
    
    def _close_position(self, candle: pd.Series, reason: str):
        """
        Cierra la posición actual.
        
        Args:
            candle: Datos de la vela actual
            reason: Razón del cierre
        """
        if not self.current_position:
            return
        
        exit_price = candle['close']
        entry_price = self.current_position['entry_price']
        quantity = self.current_position['quantity']
        
        # Calcular P&L
        if self.current_position['direction'] == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
        
        # Actualizar capital
        self.current_capital = quantity * exit_price
        
        # Registrar operación
        trade = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': candle['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': self.current_position['direction'],
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': (pnl / (entry_price * quantity)) * 100,
            'entry_reason': self.current_position['reason'],
            'exit_reason': reason
        }
        
        self.trades.append(trade)
        
        print(f"🔴 Salida {self.current_position['direction']}: {exit_price:.2f} - {reason}")
        print(f"💰 P&L: ${pnl:.2f} ({trade['return_pct']:.2f}%)")
        
        # Limpiar posición
        self.current_position = None
    
    def _calculate_portfolio_value(self, candle: pd.Series) -> float:
        """
        Calcula el valor actual del portafolio.
        
        Args:
            candle: Datos de la vela actual
            
        Returns:
            Valor del portafolio
        """
        if not self.current_position:
            return self.current_capital
        
        # Valor de la posición actual
        quantity = self.current_position['quantity']
        current_price = candle['close']
        
        return quantity * current_price
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento del backtesting.
        
        Returns:
            Diccionario con métricas
        """
        if not self.trades:
            return self._get_empty_results("No se ejecutaron operaciones")
        
        # Métricas básicas
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # Ganancia/Pérdida total
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown máximo
        max_drawdown = self._calculate_max_drawdown()
        
        # Retorno total
        final_value = self.portfolio_value[-1] if self.portfolio_value else self.initial_capital
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_value': final_value,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'trades_detail': self.trades,
            'portfolio_curve': self.portfolio_value
        }
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calcula el drawdown máximo del portafolio.
        
        Returns:
            Drawdown máximo como porcentaje
        """
        if not self.portfolio_value:
            return 0.0
        
        portfolio_series = pd.Series(self.portfolio_value)
        peak = portfolio_series.expanding(min_periods=1).max()
        drawdown = (portfolio_series - peak) / peak * 100
        
        return abs(drawdown.min())
    
    def _calculate_rsi_manual(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI manually when pandas_ta is not available.
        
        Args:
            prices: Series of closing prices
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_manual(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, pd.Series]:
        """
        Calculate MACD manually when pandas_ta is not available.
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            Dictionary with MACD, Signal, and Histogram series
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def _get_empty_results(self, message: str) -> Dict[str, Any]:
        """
        Retorna resultados vacíos con mensaje de error.
        
        Args:
            message: Mensaje de error
            
        Returns:
            Diccionario con resultados vacíos
        """
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'final_value': self.initial_capital,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'error': message,
            'trades_detail': [],
            'portfolio_curve': []
        }

# Función principal de conveniencia
def run_backtest(strategy: Dict[str, Any], historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Función principal para ejecutar un backtest.
    
    Args:
        strategy: Diccionario con parámetros de la estrategia
        historical_data: DataFrame con datos OHLCV
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    backtester = BacktesterEngine()
    return backtester.run_backtest(strategy, historical_data)

# Función de ejemplo para demostrar uso
async def demo_backtest():
    """
    Función de demostración del backtester.
    """
    from datanexus_connector import DataNexus
    
    print("🎯 DEMO: Backtesting con datos reales")
    print("="*50)
    
    # Inicializar DataNexus
    data_nexus = DataNexus()
    
    # Obtener datos históricos
    print("📊 Obteniendo datos históricos de BTC-USDT...")
    historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 500)
    
    if historical_data is None:
        print("❌ Error: No se pudieron obtener datos históricos")
        return
    
    print(f"✅ Datos obtenidos: {len(historical_data)} velas")
    
    # Definir estrategia RSI
    strategy = {
        'indicator': 'RSI',
        'entry_level': 30,
        'exit_level': 70,
        'rsi_period': 14
    }
    
    # Ejecutar backtest
    print("\n🚀 Ejecutando backtest...")
    results = run_backtest(strategy, historical_data)
    
    # Mostrar resultados
    print("\n📊 RESULTADOS DEL BACKTEST:")
    print("="*50)
    print(f"📈 Operaciones totales: {results['total_trades']}")
    print(f"🏆 Operaciones ganadoras: {results['winning_trades']}")
    print(f"❌ Operaciones perdedoras: {results['losing_trades']}")
    print(f"🎯 Tasa de éxito: {results['win_rate']:.2f}%")
    print(f"💰 Factor de ganancia: {results['profit_factor']:.2f}")
    print(f"📉 Drawdown máximo: {results['max_drawdown']:.2f}%")
    print(f"💵 Retorno total: {results['total_return']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_backtest())