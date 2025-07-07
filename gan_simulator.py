
#!/usr/bin/env python3
"""
GAN Market Simulator for DataNexus - Synthetic Market Data Generation
Este módulo utiliza Generative Adversarial Networks para generar datos de mercado sintéticos
para pruebas de estrés avanzadas de estrategias de trading.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pickle
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeGAN:
    """
    Implementación de TimeGAN (Time-series Generative Adversarial Network)
    para generar datos de mercado sintéticos realistas.
    """
    
    def __init__(self, sequence_length: int = 100, feature_dim: int = 5, 
                 hidden_dim: int = 64, num_layers: int = 3):
        """
        Inicializa el modelo TimeGAN.
        
        Args:
            sequence_length: Longitud de las secuencias temporales
            feature_dim: Número de características (OHLCV = 5)
            hidden_dim: Dimensión de las capas ocultas
            num_layers: Número de capas LSTM
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaler = MinMaxScaler()
        
        # Modelos de la arquitectura TimeGAN
        self.generator = None
        self.discriminator = None
        self.embedder = None
        self.recovery = None
        
        # Historial de entrenamiento
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'e_loss': [],
            'r_loss': []
        }
        
        logger.info(f"TimeGAN inicializado: seq_len={sequence_length}, features={feature_dim}")
    
    def _build_embedder(self) -> keras.Model:
        """
        Construye la red embedder que convierte datos reales a espacio latente.
        
        Returns:
            Modelo embedder
        """
        model = keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_dim, return_sequences=True),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(self.hidden_dim, activation='sigmoid'))
        ], name='embedder')
        
        return model
    
    def _build_recovery(self) -> keras.Model:
        """
        Construye la red recovery que convierte del espacio latente a datos originales.
        
        Returns:
            Modelo recovery
        """
        model = keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.sequence_length, self.hidden_dim)),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_dim, return_sequences=True),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(self.feature_dim, activation='sigmoid'))
        ], name='recovery')
        
        return model
    
    def _build_generator(self) -> keras.Model:
        """
        Construye la red generadora que crea datos sintéticos en espacio latente.
        
        Returns:
            Modelo generator
        """
        model = keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.sequence_length, self.hidden_dim)),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_dim, return_sequences=True),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(self.hidden_dim, activation='sigmoid'))
        ], name='generator')
        
        return model
    
    def _build_discriminator(self) -> keras.Model:
        """
        Construye la red discriminadora que distingue entre datos reales y sintéticos.
        
        Returns:
            Modelo discriminator
        """
        model = keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.sequence_length, self.hidden_dim)),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_dim),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        return model
    
    def _prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Datos normalizados y secuenciados
        """
        # Seleccionar columnas OHLCV
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in ohlcv_columns):
            raise ValueError(f"Datos deben contener columnas: {ohlcv_columns}")
        
        # Normalizar datos
        normalized_data = self.scaler.fit_transform(data[ohlcv_columns])
        
        # Crear secuencias
        sequences = []
        for i in range(len(normalized_data) - self.sequence_length + 1):
            sequences.append(normalized_data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train(self, historical_data: pd.DataFrame, epochs: int = 100, 
              batch_size: int = 32, save_path: str = 'models/timegan_model.pkl') -> Dict[str, Any]:
        """
        Entrena el modelo TimeGAN con datos históricos.
        
        Args:
            historical_data: DataFrame con datos históricos OHLCV
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            save_path: Ruta para guardar el modelo
            
        Returns:
            Métricas de entrenamiento
        """
        logger.info(f"🚀 Iniciando entrenamiento TimeGAN con {len(historical_data)} muestras")
        logger.info(f"📊 Parámetros: epochs={epochs}, batch_size={batch_size}")
        
        try:
            # Preparar datos
            sequences = self._prepare_data(historical_data)
            logger.info(f"✅ Datos preparados: {sequences.shape[0]} secuencias de {sequences.shape[1]} pasos")
            
            # Construir modelos
            self.embedder = self._build_embedder()
            self.recovery = self._build_recovery()
            self.generator = self._build_generator()
            self.discriminator = self._build_discriminator()
            
            # Optimizadores
            optimizer_g = keras.optimizers.Adam(learning_rate=0.001)
            optimizer_d = keras.optimizers.Adam(learning_rate=0.001)
            optimizer_e = keras.optimizers.Adam(learning_rate=0.001)
            optimizer_r = keras.optimizers.Adam(learning_rate=0.001)
            
            # Entrenamiento principal
            for epoch in range(epochs):
                epoch_g_loss = 0
                epoch_d_loss = 0
                epoch_e_loss = 0
                epoch_r_loss = 0
                
                # Mezclar datos
                indices = np.random.permutation(len(sequences))
                
                for i in range(0, len(sequences), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    real_batch = sequences[batch_indices]
                    
                    # Fase 1: Entrenar Embedder y Recovery
                    with tf.GradientTape() as tape:
                        embedded = self.embedder(real_batch)
                        recovered = self.recovery(embedded)
                        reconstruction_loss = tf.reduce_mean(tf.square(real_batch - recovered))
                    
                    # Actualizar embedder y recovery
                    grads_e = tape.gradient(reconstruction_loss, self.embedder.trainable_variables)
                    grads_r = tape.gradient(reconstruction_loss, self.recovery.trainable_variables)
                    optimizer_e.apply_gradients(zip(grads_e, self.embedder.trainable_variables))
                    optimizer_r.apply_gradients(zip(grads_r, self.recovery.trainable_variables))
                    
                    epoch_e_loss += reconstruction_loss.numpy()
                    epoch_r_loss += reconstruction_loss.numpy()
                    
                    # Fase 2: Entrenar Discriminador
                    with tf.GradientTape() as tape:
                        # Datos reales embedidos
                        real_embedded = self.embedder(real_batch)
                        
                        # Generar ruido y datos sintéticos
                        noise = tf.random.normal([len(real_batch), self.sequence_length, self.hidden_dim])
                        fake_embedded = self.generator(noise)
                        
                        # Predicciones del discriminador
                        real_pred = self.discriminator(real_embedded)
                        fake_pred = self.discriminator(fake_embedded)
                        
                        # Pérdida del discriminador
                        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(real_pred), logits=real_pred))
                        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.zeros_like(fake_pred), logits=fake_pred))
                        d_loss = d_loss_real + d_loss_fake
                    
                    # Actualizar discriminador
                    grads_d = tape.gradient(d_loss, self.discriminator.trainable_variables)
                    optimizer_d.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
                    
                    epoch_d_loss += d_loss.numpy()
                    
                    # Fase 3: Entrenar Generador
                    with tf.GradientTape() as tape:
                        noise = tf.random.normal([len(real_batch), self.sequence_length, self.hidden_dim])
                        fake_embedded = self.generator(noise)
                        fake_pred = self.discriminator(fake_embedded)
                        
                        # Pérdida del generador (engañar al discriminador)
                        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(fake_pred), logits=fake_pred))
                    
                    # Actualizar generador
                    grads_g = tape.gradient(g_loss, self.generator.trainable_variables)
                    optimizer_g.apply_gradients(zip(grads_g, self.generator.trainable_variables))
                    
                    epoch_g_loss += g_loss.numpy()
                
                # Promediar pérdidas por época
                num_batches = len(sequences) // batch_size
                avg_g_loss = epoch_g_loss / num_batches
                avg_d_loss = epoch_d_loss / num_batches
                avg_e_loss = epoch_e_loss / num_batches
                avg_r_loss = epoch_r_loss / num_batches
                
                # Guardar historial
                self.training_history['g_loss'].append(avg_g_loss)
                self.training_history['d_loss'].append(avg_d_loss)
                self.training_history['e_loss'].append(avg_e_loss)
                self.training_history['r_loss'].append(avg_r_loss)
                
                # Log progreso
                if epoch % 10 == 0:
                    logger.info(f"Época {epoch}/{epochs}: "
                               f"G_Loss: {avg_g_loss:.4f}, "
                               f"D_Loss: {avg_d_loss:.4f}, "
                               f"E_Loss: {avg_e_loss:.4f}, "
                               f"R_Loss: {avg_r_loss:.4f}")
            
            # Guardar modelo
            self._save_model(save_path)
            
            logger.info("🎉 Entrenamiento TimeGAN completado exitosamente")
            
            return {
                'training_history': self.training_history,
                'final_losses': {
                    'generator': avg_g_loss,
                    'discriminator': avg_d_loss,
                    'embedder': avg_e_loss,
                    'recovery': avg_r_loss
                },
                'model_path': save_path
            }
            
        except Exception as e:
            logger.error(f"❌ Error durante entrenamiento: {str(e)}")
            raise
    
    def generate_scenarios(self, num_scenarios: int = 1000, 
                          sequence_length: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Genera escenarios sintéticos de mercado usando el modelo entrenado.
        
        Args:
            num_scenarios: Número de escenarios a generar
            sequence_length: Longitud de cada escenario (usa self.sequence_length por defecto)
            
        Returns:
            Lista de DataFrames con datos sintéticos
        """
        if not all([self.generator, self.recovery]):
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        seq_len = sequence_length or self.sequence_length
        logger.info(f"🔬 Generando {num_scenarios} escenarios sintéticos de {seq_len} pasos cada uno")
        
        synthetic_scenarios = []
        
        try:
            for i in range(num_scenarios):
                # Generar ruido aleatorio
                noise = tf.random.normal([1, seq_len, self.hidden_dim])
                
                # Generar datos en espacio latente
                fake_embedded = self.generator(noise)
                
                # Recuperar a espacio original
                synthetic_data = self.recovery(fake_embedded)
                
                # Desnormalizar
                synthetic_data_np = synthetic_data.numpy().reshape(seq_len, self.feature_dim)
                denormalized_data = self.scaler.inverse_transform(synthetic_data_np)
                
                # Crear DataFrame
                df = pd.DataFrame(denormalized_data, columns=['open', 'high', 'low', 'close', 'volume'])
                
                # Añadir timestamps sintéticos
                base_time = datetime.now()
                df['timestamp'] = [base_time + timedelta(minutes=i) for i in range(seq_len)]
                
                # Validar datos (precios positivos, high >= low, etc.)
                df = self._validate_synthetic_data(df)
                
                synthetic_scenarios.append(df)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"   Generados {i + 1}/{num_scenarios} escenarios...")
            
            logger.info(f"✅ Generación completada: {len(synthetic_scenarios)} escenarios sintéticos")
            return synthetic_scenarios
            
        except Exception as e:
            logger.error(f"❌ Error generando escenarios: {str(e)}")
            raise
    
    def _validate_synthetic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y corrige datos sintéticos para asegurar coherencia financiera.
        
        Args:
            df: DataFrame con datos sintéticos
            
        Returns:
            DataFrame validado y corregido
        """
        # Asegurar precios positivos
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = np.maximum(df[col], 0.01)  # Mínimo 0.01
        
        # Asegurar high >= max(open, close) y low <= min(open, close)
        for i in range(len(df)):
            open_price = df.iloc[i]['open']
            close_price = df.iloc[i]['close']
            
            # Corregir high
            min_high = max(open_price, close_price)
            if df.iloc[i]['high'] < min_high:
                df.iloc[i, df.columns.get_loc('high')] = min_high * 1.001  # Pequeño margen
            
            # Corregir low
            max_low = min(open_price, close_price)
            if df.iloc[i]['low'] > max_low:
                df.iloc[i, df.columns.get_loc('low')] = max_low * 0.999  # Pequeño margen
        
        # Asegurar volumen positivo
        df['volume'] = np.maximum(df['volume'], 1.0)
        
        return df
    
    def _save_model(self, save_path: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            save_path: Ruta donde guardar el modelo
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            model_data = {
                'generator': self.generator,
                'discriminator': self.discriminator,
                'embedder': self.embedder,
                'recovery': self.recovery,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'feature_dim': self.feature_dim,
                'hidden_dim': self.hidden_dim,
                'training_history': self.training_history
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Modelo guardado en: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {str(e)}")
    
    def load_model(self, load_path: str):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            load_path: Ruta del modelo a cargar
        """
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.generator = model_data['generator']
            self.discriminator = model_data['discriminator']
            self.embedder = model_data['embedder']
            self.recovery = model_data['recovery']
            self.scaler = model_data['scaler']
            self.sequence_length = model_data['sequence_length']
            self.feature_dim = model_data['feature_dim']
            self.hidden_dim = model_data['hidden_dim']
            self.training_history = model_data['training_history']
            
            logger.info(f"📂 Modelo cargado desde: {load_path}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def plot_training_history(self):
        """
        Visualiza el historial de entrenamiento.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator Loss
        axes[0, 0].plot(self.training_history['g_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Discriminator Loss
        axes[0, 1].plot(self.training_history['d_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Embedder Loss
        axes[1, 0].plot(self.training_history['e_loss'])
        axes[1, 0].set_title('Embedder Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        
        # Recovery Loss
        axes[1, 1].plot(self.training_history['r_loss'])
        axes[1, 1].set_title('Recovery Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('timegan_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class GANMarketSimulator:
    """
    Simulador principal que coordina el entrenamiento y generación de datos sintéticos.
    """
    
    def __init__(self):
        """Inicializa el simulador de mercados GAN."""
        self.timegan = None
        self.synthetic_scenarios = []
        self.model_path = 'models/timegan_model.pkl'
        
        logger.info("🤖 GANMarketSimulator inicializado")
    
    def train_market_gan(self, historical_data: pd.DataFrame, 
                        sequence_length: int = 200, epochs: int = 100) -> Dict[str, Any]:
        """
        Entrena un modelo TimeGAN con datos históricos masivos.
        
        Args:
            historical_data: DataFrame con datos históricos OHLCV
            sequence_length: Longitud de las secuencias para entrenamiento
            epochs: Número de épocas de entrenamiento
            
        Returns:
            Métricas de entrenamiento
        """
        logger.info("🧠 Iniciando entrenamiento del GAN de mercados sintéticos")
        logger.info(f"📊 Datos de entrada: {len(historical_data)} registros")
        logger.info(f"⚙️ Configuración: seq_length={sequence_length}, epochs={epochs}")
        
        try:
            # Inicializar TimeGAN
            self.timegan = TimeGAN(
                sequence_length=sequence_length,
                feature_dim=5,  # OHLCV
                hidden_dim=128,
                num_layers=3
            )
            
            # Entrenar modelo
            training_results = self.timegan.train(
                historical_data=historical_data,
                epochs=epochs,
                batch_size=64,
                save_path=self.model_path
            )
            
            logger.info("🎉 Entrenamiento del GAN completado exitosamente")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento del GAN: {str(e)}")
            raise
    
    def generate_scenarios(self, num_scenarios: int = 1000, 
                          scenario_length: int = 200) -> List[pd.DataFrame]:
        """
        Genera múltiples escenarios sintéticos de mercado.
        
        Args:
            num_scenarios: Número de escenarios a generar
            scenario_length: Longitud de cada escenario
            
        Returns:
            Lista de DataFrames con escenarios sintéticos
        """
        logger.info(f"🎲 Generando {num_scenarios} escenarios sintéticos")
        
        try:
            # Cargar modelo si no está en memoria
            if self.timegan is None:
                if os.path.exists(self.model_path):
                    self.timegan = TimeGAN()
                    self.timegan.load_model(self.model_path)
                else:
                    raise ValueError("Modelo no encontrado. Entrenar primero con train_market_gan()")
            
            # Generar escenarios
            self.synthetic_scenarios = self.timegan.generate_scenarios(
                num_scenarios=num_scenarios,
                sequence_length=scenario_length
            )
            
            logger.info(f"✅ Generados {len(self.synthetic_scenarios)} escenarios sintéticos")
            return self.synthetic_scenarios
            
        except Exception as e:
            logger.error(f"❌ Error generando escenarios: {str(e)}")
            raise
    
    def stress_test_strategy(self, strategy: Dict[str, Any], 
                           synthetic_scenarios: Optional[List[pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Realiza prueba de estrés de una estrategia contra escenarios sintéticos.
        
        Args:
            strategy: Diccionario con parámetros de la estrategia
            synthetic_scenarios: Lista de escenarios sintéticos (usa los generados si es None)
            
        Returns:
            Resultados de la prueba de estrés
        """
        from backtester import run_backtest
        
        scenarios = synthetic_scenarios or self.synthetic_scenarios
        if not scenarios:
            raise ValueError("No hay escenarios sintéticos disponibles")
        
        logger.info(f"🧪 Iniciando prueba de estrés con {len(scenarios)} escenarios sintéticos")
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, scenario in enumerate(scenarios):
            try:
                # Ejecutar backtest en escenario sintético
                backtest_result = run_backtest(strategy, scenario)
                
                if 'error' not in backtest_result:
                    results.append(backtest_result)
                    successful_tests += 1
                else:
                    failed_tests += 1
                
                # Log progreso cada 100 escenarios
                if (i + 1) % 100 == 0:
                    logger.info(f"   Procesados {i + 1}/{len(scenarios)} escenarios...")
                    
            except Exception as e:
                failed_tests += 1
                logger.warning(f"Error en escenario {i}: {str(e)}")
        
        # Calcular estadísticas agregadas
        if results:
            win_rates = [r['win_rate'] for r in results]
            profit_factors = [r['profit_factor'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            total_returns = [r['total_return'] for r in results]
            
            stress_test_results = {
                'total_scenarios': len(scenarios),
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / len(scenarios),
                'win_rate_stats': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'min': np.min(win_rates),
                    'max': np.max(win_rates),
                    'percentile_25': np.percentile(win_rates, 25),
                    'percentile_75': np.percentile(win_rates, 75)
                },
                'profit_factor_stats': {
                    'mean': np.mean(profit_factors),
                    'std': np.std(profit_factors),
                    'min': np.min(profit_factors),
                    'max': np.max(profit_factors)
                },
                'max_drawdown_stats': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns)
                },
                'total_return_stats': {
                    'mean': np.mean(total_returns),
                    'std': np.std(total_returns),
                    'min': np.min(total_returns),
                    'max': np.max(total_returns)
                },
                'robustness_score': self._calculate_robustness_score(results)
            }
        else:
            stress_test_results = {
                'error': 'Ningún escenario fue procesado exitosamente',
                'total_scenarios': len(scenarios),
                'successful_tests': 0,
                'failed_tests': failed_tests
            }
        
        logger.info(f"🎯 Prueba de estrés completada: {successful_tests}/{len(scenarios)} éxitos")
        return stress_test_results
    
    def _calculate_robustness_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calcula un score de robustez basado en los resultados de las pruebas.
        
        Args:
            results: Lista de resultados de backtest
            
        Returns:
            Score de robustez (0-100)
        """
        if not results:
            return 0.0
        
        # Métricas para calcular robustez
        profitable_scenarios = sum(1 for r in results if r['total_return'] > 0)
        profitability_rate = profitable_scenarios / len(results)
        
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in results])
        
        # Score compuesto (0-100)
        robustness_score = (
            profitability_rate * 40 +  # 40% peso a escenarios rentables
            min(avg_win_rate / 100, 1.0) * 30 +  # 30% peso a win rate
            min(avg_profit_factor / 3, 1.0) * 20 +  # 20% peso a profit factor
            max(0, 1 - avg_max_drawdown / 50) * 10  # 10% peso a drawdown bajo
        ) * 100
        
        return min(max(robustness_score, 0), 100)


# Funciones de conveniencia para integración
async def train_market_gan(historical_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Función asíncrona para entrenar GAN de mercado.
    
    Args:
        historical_data: Datos históricos para entrenamiento
        **kwargs: Argumentos adicionales para el entrenamiento
        
    Returns:
        Resultados del entrenamiento
    """
    simulator = GANMarketSimulator()
    return simulator.train_market_gan(historical_data, **kwargs)


def generate_scenarios(generator_model=None, num_scenarios: int = 1000, **kwargs) -> List[pd.DataFrame]:
    """
    Función para generar escenarios sintéticos.
    
    Args:
        generator_model: Modelo generador (ignorado, usa modelo guardado)
        num_scenarios: Número de escenarios a generar
        **kwargs: Argumentos adicionales
        
    Returns:
        Lista de escenarios sintéticos
    """
    simulator = GANMarketSimulator()
    return simulator.generate_scenarios(num_scenarios, **kwargs)


def stress_test_strategy(strategy: Dict[str, Any], num_scenarios: int = 1000) -> Dict[str, Any]:
    """
    Función para realizar prueba de estrés de estrategia.
    
    Args:
        strategy: Estrategia a probar
        num_scenarios: Número de escenarios para la prueba
        
    Returns:
        Resultados de la prueba de estrés
    """
    simulator = GANMarketSimulator()
    
    # Generar escenarios si no existen
    if not simulator.synthetic_scenarios:
        simulator.generate_scenarios(num_scenarios)
    
    return simulator.stress_test_strategy(strategy)


# Función de demostración
async def demo_gan_simulator():
    """
    Demostración del simulador GAN.
    """
    from datanexus_connector import DataNexus
    
    print("🚀 DEMO: Simulador de Mercados Sintéticos con GAN")
    print("="*60)
    
    try:
        # Obtener datos históricos
        data_nexus = DataNexus()
        print("📊 Obteniendo datos históricos...")
        historical_data = await data_nexus.get_crypto_data('BTCUSDT', '1h', 1000)
        
        if historical_data is None:
            print("❌ No se pudieron obtener datos históricos")
            return
        
        print(f"✅ Datos obtenidos: {len(historical_data)} registros")
        
        # Crear simulador
        simulator = GANMarketSimulator()
        
        # Entrenar GAN (versión reducida para demo)
        print("\n🧠 Entrenando TimeGAN (versión demo rápida)...")
        training_results = simulator.train_market_gan(
            historical_data=historical_data,
            sequence_length=50,  # Reducido para demo
            epochs=20  # Reducido para demo
        )
        
        print("🎉 Entrenamiento completado")
        
        # Generar escenarios sintéticos
        print("\n🎲 Generando escenarios sintéticos...")
        synthetic_scenarios = simulator.generate_scenarios(
            num_scenarios=10,  # Reducido para demo
            scenario_length=50
        )
        
        print(f"✅ Generados {len(synthetic_scenarios)} escenarios sintéticos")
        
        # Prueba de estrés con estrategia simple
        simple_strategy = {
            'indicator': 'RSI',
            'entry_level': 30,
            'exit_level': 70,
            'rsi_period': 14
        }
        
        print("\n🧪 Realizando prueba de estrés...")
        stress_results = simulator.stress_test_strategy(simple_strategy)
        
        print("📊 RESULTADOS DE PRUEBA DE ESTRÉS:")
        print(f"   Escenarios exitosos: {stress_results['successful_tests']}/{stress_results['total_scenarios']}")
        if 'robustness_score' in stress_results:
            print(f"   Score de robustez: {stress_results['robustness_score']:.2f}/100")
            print(f"   Win rate promedio: {stress_results['win_rate_stats']['mean']:.2f}%")
            print(f"   Profit factor promedio: {stress_results['profit_factor_stats']['mean']:.2f}")
        
        print("\n🎯 Demo completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error en demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_gan_simulator())
