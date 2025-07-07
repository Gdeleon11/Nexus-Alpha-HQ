
#!/usr/bin/env python3
"""
Causal Analysis Microservice
Microservicio independiente para análisis de inferencia causal
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

try:
    from econml.dml import LinearDML
    from econml.dr import DRLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

app = Flask(__name__)

class CausalEngine:
    """
    Motor de inferencia causal para analizar relaciones causa-efecto en mercados financieros.
    """
    
    def __init__(self):
        self.available_methods = []
        
        if DOWHY_AVAILABLE:
            self.available_methods.append('dowhy')
        if ECONML_AVAILABLE:
            self.available_methods.append('econml')
        
        if not self.available_methods:
            self.available_methods.append('fallback')
        
        print(f"CausalEngine initialized with methods: {self.available_methods}")
    
    def estimate_causal_effect(self, event_data: Dict[str, Any], market_data_dict: Dict[str, Any]) -> float:
        """
        Estima el efecto causal de un evento sobre los datos del mercado.
        """
        try:
            # Convertir diccionario a DataFrame
            market_data = pd.DataFrame(market_data_dict)
            
            # Preprocesar los datos para el análisis causal
            processed_data = self._prepare_causal_data(event_data, market_data)
            
            if processed_data is None or len(processed_data) < 10:
                return 0.1
            
            # Seleccionar método de análisis causal
            if 'dowhy' in self.available_methods:
                causal_score = self._estimate_with_dowhy(processed_data, event_data)
            elif 'econml' in self.available_methods:
                causal_score = self._estimate_with_econml(processed_data, event_data)
            else:
                causal_score = self._estimate_with_fallback(processed_data, event_data)
            
            # Normalizar el score entre 0 y 1
            causal_score = max(0.0, min(1.0, causal_score))
            
            return round(causal_score, 3)
            
        except Exception as e:
            print(f"Error en análisis causal: {e}")
            return 0.1
    
    def _prepare_causal_data(self, event_data: Dict[str, Any], market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepara los datos para el análisis causal.
        """
        try:
            if market_data.empty or len(market_data) < 5:
                return None
            
            df = market_data.copy()
            
            # Asegurar que timestamp sea datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
            
            # Calcular variables de resultado
            df['price_change'] = df['close'].pct_change()
            df['price_volatility'] = df['price_change'].rolling(window=5).std()
            df['volume_change'] = df['volume'].pct_change() if 'volume' in df.columns else 0
            
            # Crear variable de tratamiento
            event_timestamp = event_data.get('timestamp')
            if event_timestamp:
                df['treatment'] = (df['timestamp'] >= pd.to_datetime(event_timestamp)).astype(int)
            else:
                midpoint = len(df) // 2
                df['treatment'] = 0
                df.loc[midpoint:, 'treatment'] = 1
            
            # Crear variables de control
            df['lagged_price'] = df['close'].shift(1)
            df['moving_avg_5'] = df['close'].rolling(window=5).mean()
            df['rsi'] = self._calculate_simple_rsi(df['close'], 14)
            
            # Agregar intensidad y sentimiento del evento
            event_intensity = event_data.get('intensity', 0.5)
            df['event_intensity'] = event_intensity * df['treatment']
            
            sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
            sentiment_score = sentiment_mapping.get(event_data.get('sentiment', 'neutral'), 0)
            df['event_sentiment'] = sentiment_score * df['treatment']
            
            df = df.dropna()
            return df
            
        except Exception as e:
            print(f"Error preparando datos causales: {e}")
            return None
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula RSI simple.
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _estimate_with_dowhy(self, data: pd.DataFrame, event_data: Dict[str, Any]) -> float:
        """
        Estima efecto causal usando DoWhy.
        """
        try:
            causal_graph = """
            digraph {
                treatment -> price_change;
                event_intensity -> price_change;
                event_sentiment -> price_change;
                lagged_price -> price_change;
                moving_avg_5 -> price_change;
                rsi -> price_change;
                lagged_price -> treatment;
                moving_avg_5 -> treatment;
            }
            """
            
            model = CausalModel(
                data=data,
                treatment='treatment',
                outcome='price_change',
                graph=causal_graph
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            estimate_iv = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            causal_effect = abs(estimate_iv.value)
            confidence_score = min(1.0, causal_effect * 10)
            
            try:
                refutation = model.refute_estimate(
                    identified_estimand, 
                    estimate_iv,
                    method_name="random_common_cause"
                )
                
                if hasattr(refutation, 'new_effect'):
                    robustness_factor = 1 - abs(estimate_iv.value - refutation.new_effect) / max(abs(estimate_iv.value), 0.01)
                    confidence_score *= max(0.1, robustness_factor)
            except:
                confidence_score *= 0.8
            
            return confidence_score
            
        except Exception as e:
            print(f"Error en DoWhy: {e}")
            return self._estimate_with_fallback(data, event_data)
    
    def _estimate_with_econml(self, data: pd.DataFrame, event_data: Dict[str, Any]) -> float:
        """
        Estima efecto causal usando EconML.
        """
        try:
            X = data[['lagged_price', 'moving_avg_5', 'rsi']].fillna(0)
            T = data['treatment']
            Y = data['price_change'].fillna(0)
            
            if len(X) < 10 or T.sum() == 0 or T.sum() == len(T):
                return self._estimate_with_fallback(data, event_data)
            
            est = LinearDML(model_y='auto', model_t='auto', random_state=42)
            est.fit(Y=Y, T=T, X=X)
            
            treatment_effect = est.effect(X)
            avg_effect = np.mean(np.abs(treatment_effect))
            
            confidence_score = min(1.0, avg_effect * 5)
            
            try:
                conf_int = est.effect_interval(X, alpha=0.1)
                if np.all(conf_int[0] > 0) or np.all(conf_int[1] < 0):
                    confidence_score *= 1.2
            except:
                pass
            
            return min(1.0, confidence_score)
            
        except Exception as e:
            print(f"Error en EconML: {e}")
            return self._estimate_with_fallback(data, event_data)
    
    def _estimate_with_fallback(self, data: pd.DataFrame, event_data: Dict[str, Any]) -> float:
        """
        Método de fallback para estimar causalidad.
        """
        try:
            treated = data[data['treatment'] == 1]['price_change']
            control = data[data['treatment'] == 0]['price_change']
            
            if len(treated) == 0 or len(control) == 0:
                return 0.1
            
            mean_diff = abs(treated.mean() - control.mean())
            
            correlation = abs(data['event_intensity'].corr(data['price_change']))
            if pd.isna(correlation):
                correlation = 0
            
            vol_treated = treated.std() if len(treated) > 1 else 0
            vol_control = control.std() if len(control) > 1 else 0
            vol_ratio = min(vol_treated / max(vol_control, 0.001), vol_control / max(vol_treated, 0.001))
            
            causal_score = (mean_diff * 10 + correlation + (1 - vol_ratio)) / 3
            
            sentiment_boost = 1.0
            if event_data.get('sentiment') in ['positive', 'negative']:
                sentiment_boost = 1.2
            
            causal_score *= sentiment_boost
            
            return min(1.0, causal_score)
            
        except Exception as e:
            print(f"Error en análisis fallback: {e}")
            return 0.1

# Inicializar el motor causal
causal_engine = CausalEngine()

@app.route('/analyze', methods=['POST'])
def analyze_causal_effect():
    """
    Endpoint para análisis de inferencia causal.
    """
    try:
        data = request.get_json()
        
        if not data or 'event_data' not in data or 'market_data' not in data:
            return jsonify({
                'error': 'Faltan datos requeridos: event_data y market_data'
            }), 400
        
        event_data = data['event_data']
        market_data = data['market_data']
        
        # Realizar análisis causal
        causal_score = causal_engine.estimate_causal_effect(event_data, market_data)
        
        return jsonify({
            'causal_score': causal_score,
            'interpretation': (
                'Alta causalidad detectada' if causal_score > 0.7 else
                'Causalidad moderada detectada' if causal_score > 0.4 else
                'Baja causalidad detectada' if causal_score > 0.2 else
                'Causalidad insignificante'
            ),
            'available_methods': causal_engine.available_methods
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error en análisis causal: {str(e)}',
            'causal_score': 0.1
        }), 500

@app.route('/')
def home():
    """
    Ruta principal del microservicio.
    """
    return jsonify({
        'service': 'causal-analysis-microservice',
        'status': 'running',
        'available_endpoints': ['/analyze', '/health'],
        'available_methods': causal_engine.available_methods
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de verificación de salud del servicio.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'causal-analysis-microservice',
        'available_methods': causal_engine.available_methods
    })

@app.route('/api/log_override', methods=['POST'])
def log_override():
    """
    Endpoint para registrar overrides del operador (RLHF).
    """
    try:
        data = request.get_json()
        signal_id = data.get('signal_id')
        reason = data.get('reason')
        
        if not signal_id or not reason:
            return jsonify({'error': 'signal_id y reason son requeridos'}), 400
        
        # Conectar a la base de datos
        import sqlite3
        from datetime import datetime
        
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        
        # Crear tabla de overrides si no existe
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                asset TEXT,
                timeframe TEXT,
                trigger_reason TEXT
            )
        ''')
        
        # Obtener información adicional de la señal
        cursor.execute('SELECT asset, verdict FROM trades WHERE signal_id = ?', (signal_id,))
        signal_info = cursor.fetchone()
        
        asset = signal_info[0] if signal_info else 'UNKNOWN'
        
        # Obtener trigger_reason del signal_id (si está disponible)
        # Por ahora usaremos un placeholder, en una implementación completa 
        # podríamos almacenar más detalles de la señal
        trigger_reason = 'RSI Signal'  # Placeholder
        
        # Insertar override
        cursor.execute('''
            INSERT INTO overrides (signal_id, reason, timestamp, asset, timeframe, trigger_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (signal_id, reason, datetime.now().isoformat(), asset, '5m', trigger_reason))
        
        conn.commit()
        conn.close()
        
        print(f"🧠 Override registrado: {reason} para {signal_id[:8]}... ({asset})")
        
        return jsonify({
            'status': 'success',
            'message': 'Override registrado correctamente',
            'signal_id': signal_id,
            'reason': reason
        })
        
    except Exception as e:
        print(f"Error registrando override: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/update_signal_result', methods=['POST'])
def update_signal_result():
    """
    Endpoint para actualizar el resultado de una señal (WIN/LOSS).
    """
    try:
        data = request.get_json()
        signal_id = data.get('signal_id')
        result = data.get('result')
        
        if not signal_id or result not in ['WIN', 'LOSS']:
            return jsonify({'error': 'signal_id y result (WIN/LOSS) son requeridos'}), 400
        
        import sqlite3
        from datetime import datetime
        
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        
        # Actualizar resultado de la señal
        cursor.execute('''
            UPDATE trades 
            SET result = ?, updated_at = ?
            WHERE signal_id = ?
        ''', (result, datetime.now().isoformat(), signal_id))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Resultado actualizado: {result} para {signal_id[:8]}...")
        
        # Activar envío automático a Red de Inteligencia Colectiva
        try:
            # Obtener información de la señal para envío colectivo
            conn = sqlite3.connect('trades.db')
            cursor = conn.cursor()
            cursor.execute('SELECT asset, verdict FROM trades WHERE signal_id = ?', (signal_id,))
            trade_info = cursor.fetchone()
            conn.close()
            
            if trade_info:
                asset, verdict = trade_info
                
                # Determinar estrategia y timeframe
                estrategia_base = "RSI"
                if "MACD" in str(verdict).upper():
                    estrategia_base = "MACD"
                elif "SMA" in str(verdict).upper():
                    estrategia_base = "SMA"
                
                if "COMPRAR" in str(verdict):
                    estrategia_usada = f"{estrategia_base}_BUY"
                elif "VENDER" in str(verdict):
                    estrategia_usada = f"{estrategia_base}_SELL"
                else:
                    estrategia_usada = f"{estrategia_base}_NEUTRAL"
                
                timeframe = "5m"  # Default
                for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    if tf in str(asset) or tf in str(verdict):
                        timeframe = tf
                        break
                
                # Enviar a Red de Inteligencia Colectiva (auto-referencia)
                collective_data = {
                    'estrategia_usada': estrategia_usada,
                    'activo': asset,
                    'timeframe': timeframe,
                    'resultado_win_loss': result
                }
                
                # Llamar al endpoint colectivo directamente
                import requests
                try:
                    response = requests.post(
                        "http://0.0.0.0:5000/api/log_collective_trade",
                        json=collective_data,
                        timeout=3
                    )
                    if response.status_code == 200:
                        print(f"🌐 Enviado automáticamente a Red de Inteligencia: {estrategia_usada}")
                except:
                    print("⚠️ No se pudo enviar a Red de Inteligencia automáticamente")
        
        except Exception as e:
            print(f"⚠️ Error en envío automático: {str(e)}")
        
        return jsonify({
            'status': 'success',
            'message': 'Resultado actualizado correctamente',
            'signal_id': signal_id,
            'result': result
        })
        
    except Exception as e:
        print(f"Error actualizando resultado: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/log_collective_trade', methods=['POST'])
def log_collective_trade():
    """
    Endpoint central para recibir datos anonimizados de trades de toda la red Nexus-Alpha.
    """
    try:
        data = request.get_json()
        
        # Validar datos requeridos
        required_fields = ['estrategia_usada', 'resultado_win_loss']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        estrategia_usada = data.get('estrategia_usada')
        activo = data.get('activo', 'UNKNOWN')
        timeframe = data.get('timeframe', 'UNKNOWN')
        resultado_win_loss = data.get('resultado_win_loss')
        
        if resultado_win_loss not in ['WIN', 'LOSS']:
            return jsonify({'error': 'resultado_win_loss debe ser WIN o LOSS'}), 400
        
        import sqlite3
        from datetime import datetime
        
        conn = sqlite3.connect('collective_intelligence.db')
        cursor = conn.cursor()
        
        # Crear tabla de inteligencia colectiva si no existe
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collective_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                estrategia_usada TEXT NOT NULL,
                activo TEXT,
                timeframe TEXT,
                resultado_win_loss TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                node_hash TEXT  -- Hash anonimizado del nodo que envía
            )
        ''')
        
        # Generar hash anonimizado del nodo basado en IP y timestamp
        import hashlib
        node_identifier = request.remote_addr + str(datetime.now().date())
        node_hash = hashlib.md5(node_identifier.encode()).hexdigest()[:8]
        
        # Insertar datos anonimizados
        cursor.execute('''
            INSERT INTO collective_trades 
            (estrategia_usada, activo, timeframe, resultado_win_loss, timestamp, node_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (estrategia_usada, activo, timeframe, resultado_win_loss, 
              datetime.now().isoformat(), node_hash))
        
        conn.commit()
        conn.close()
        
        print(f"🌐 Trade colectivo registrado: {estrategia_usada} -> {resultado_win_loss} [{node_hash}]")
        
        return jsonify({
            'status': 'success',
            'message': 'Trade registrado en la red de inteligencia colectiva',
            'node_hash': node_hash
        })
        
    except Exception as e:
        print(f"Error en inteligencia colectiva: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/collective_metrics', methods=['GET'])
def get_collective_metrics():
    """
    Endpoint para obtener métricas agregadas de la red de inteligencia colectiva.
    """
    try:
        estrategia = request.args.get('estrategia')
        timeframe = request.args.get('timeframe')
        activo = request.args.get('activo')
        
        import sqlite3
        
        conn = sqlite3.connect('collective_intelligence.db')
        cursor = conn.cursor()
        
        # Construir consulta con filtros opcionales
        base_query = '''
            SELECT 
                estrategia_usada,
                timeframe,
                activo,
                COUNT(*) as total_trades,
                SUM(CASE WHEN resultado_win_loss = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN resultado_win_loss = 'LOSS' THEN 1 ELSE 0 END) as losses,
                ROUND(
                    (SUM(CASE WHEN resultado_win_loss = 'WIN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
                ) as win_rate,
                COUNT(DISTINCT node_hash) as contributing_nodes
            FROM collective_trades 
            WHERE 1=1
        '''
        
        params = []
        
        if estrategia:
            base_query += " AND UPPER(estrategia_usada) LIKE UPPER(?)"
            params.append(f'%{estrategia}%')
        
        if timeframe:
            base_query += " AND timeframe = ?"
            params.append(timeframe)
        
        if activo:
            base_query += " AND UPPER(activo) LIKE UPPER(?)"
            params.append(f'%{activo}%')
        
        base_query += '''
            GROUP BY estrategia_usada, timeframe, activo
            HAVING COUNT(*) >= 3  -- Mínimo 3 trades para considerar válido
            ORDER BY win_rate DESC, total_trades DESC
            LIMIT 20
        '''
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        # Formatear resultados
        collective_metrics = []
        for row in results:
            metric = {
                'estrategia_usada': row[0],
                'timeframe': row[1],
                'activo': row[2],
                'total_trades': row[3],
                'wins': row[4],
                'losses': row[5],
                'win_rate': row[6],
                'contributing_nodes': row[7],
                'confidence_score': min(row[3] / 10.0, 1.0)  # Confianza basada en volumen
            }
            collective_metrics.append(metric)
        
        # Estadísticas generales
        cursor.execute('SELECT COUNT(*) FROM collective_trades')
        total_collective_trades = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT node_hash) FROM collective_trades')
        total_nodes = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'collective_metrics': collective_metrics,
            'network_stats': {
                'total_trades': total_collective_trades,
                'active_nodes': total_nodes,
                'strategies_analyzed': len(collective_metrics)
            }
        })
        
    except Exception as e:
        print(f"Error obteniendo métricas colectivas: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/get_explanation/<signal_id>', methods=['GET'])
def get_explanation(signal_id):
    """
    Endpoint para obtener explicación XAI de una señal específica.
    """
    try:
        from explanation_engine import get_explanation as get_exp
        
        explanation = get_exp(signal_id)
        
        if explanation:
            return jsonify({
                'status': 'success',
                'explanation': explanation
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': f'No se encontró explicación para signal_id: {signal_id}'
            }), 404
            
    except Exception as e:
        print(f"Error obteniendo explicación: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    print("🔬 Iniciando Microservicio de Análisis Causal...")
    print(f"📊 Métodos disponibles: {causal_engine.available_methods}")
    app.run(host='0.0.0.0', port=5000, debug=True)
