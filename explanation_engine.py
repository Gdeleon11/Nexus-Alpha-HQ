
#!/usr/bin/env python3
"""
Explanation Engine for DataNexus - Explainable AI using SHAP
Este módulo proporciona explicaciones transparentes de las decisiones de trading.
"""

import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime


class ExplanationEngine:
    """
    Motor de IA Explicable que utiliza SHAP para proporcionar explicaciones
    transparentes de las decisiones de trading.
    """
    
    def __init__(self):
        """
        Inicializa el motor de explicaciones.
        """
        self.explanations = {}  # Cache de explicaciones por signal_id
        
    def generate_explanation(self, decision_inputs: Dict[str, Any], final_decision: str, signal_id: str) -> Dict[str, Any]:
        """
        Genera una explicación XAI completa para una decisión de trading.
        
        Args:
            decision_inputs: Diccionario con todos los factores de decisión
            final_decision: Decisión final tomada (COMPRAR/VENDER/NO OPERAR)
            signal_id: ID único de la señal
            
        Returns:
            Diccionario con explicación completa y datos de visualización
        """
        try:
            print(f"🔍 Generando explicación XAI para señal {signal_id[:8]}...")
            
            # Convertir inputs cualitativos a valores numéricos
            feature_values = self._convert_inputs_to_features(decision_inputs)
            feature_names = list(feature_values.keys())
            values_array = np.array(list(feature_values.values())).reshape(1, -1)
            
            # Crear modelo de decisión mock
            decision_model = self._create_decision_model()
            
            # Crear explicador SHAP
            background_data = np.zeros((10, len(feature_names)))
            explainer = shap.KernelExplainer(decision_model, background_data)
            
            # Calcular valores SHAP
            shap_values = explainer.shap_values(values_array)
            
            # Procesar resultados
            decision_index = 1 if final_decision in ['COMPRAR', 'VENDER'] else 0
            feature_impacts = shap_values[decision_index][0]
            
            # Crear explicación estructurada
            explanation_data = {
                'signal_id': signal_id,
                'decision': final_decision,
                'feature_names': feature_names,
                'feature_values': list(feature_values.values()),
                'feature_impacts': feature_impacts.tolist(),
                'base_value': explainer.expected_value[decision_index],
                'prediction_value': decision_model(values_array)[0][decision_index],
                'timestamp': datetime.now().isoformat(),
                'interpretation': self._generate_interpretation(feature_names, feature_impacts, final_decision)
            }
            
            # Generar visualizaciones
            explanation_data['force_plot'] = self._create_force_plot(explanation_data)
            explanation_data['waterfall_chart'] = self._create_waterfall_chart(explanation_data)
            explanation_data['feature_importance'] = self._create_feature_importance_chart(explanation_data)
            
            # Almacenar en cache
            self.explanations[signal_id] = explanation_data
            
            print(f"✅ Explicación XAI generada: {len(feature_names)} factores analizados")
            
            return explanation_data
            
        except Exception as e:
            print(f"⚠️ Error generando explicación XAI: {str(e)}")
            
            # Fallback simple
            return {
                'signal_id': signal_id,
                'decision': final_decision,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'interpretation': f"Error generando explicación para decisión: {final_decision}"
            }
    
    def _convert_inputs_to_features(self, decision_inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Convierte los inputs de decisión a valores numéricos para SHAP.
        
        Args:
            decision_inputs: Inputs originales de la decisión
            
        Returns:
            Diccionario con valores numéricos normalizados
        """
        features = {}
        
        # Factor 1: RSI Trigger
        trigger = decision_inputs.get('trigger', '')
        if 'sobreventa' in trigger.lower():
            features['RSI_Signal'] = -0.8
        elif 'sobrecompra' in trigger.lower():
            features['RSI_Signal'] = 0.8
        else:
            features['RSI_Signal'] = 0.0
        
        # Factor 2: Sentimiento Cognitivo (Gemini)
        cognitive = decision_inputs.get('cognitive', '')
        if 'muy positivo' in cognitive.lower():
            features['News_Sentiment'] = 1.0
        elif 'positivo' in cognitive.lower():
            features['News_Sentiment'] = 0.5
        elif 'negativo' in cognitive.lower():
            features['News_Sentiment'] = -0.5
        elif 'muy negativo' in cognitive.lower():
            features['News_Sentiment'] = -1.0
        else:
            features['News_Sentiment'] = 0.0
        
        # Factor 3: Predicción TimeGPT
        predictive = decision_inputs.get('predictive', {})
        prediction = predictive.get('prediction', 'Plana')
        confidence = predictive.get('confidence', 0.5)
        
        if prediction == 'Alcista':
            features['AI_Forecast'] = confidence
        elif prediction == 'Bajista':
            features['AI_Forecast'] = -confidence
        else:
            features['AI_Forecast'] = 0.0
        
        # Factor 4: Contexto Macroeconómico (Claude)
        macro = decision_inputs.get('macro_context', '')
        if 'positivo' in macro.lower():
            features['Macro_Context'] = 0.6
        elif 'negativo' in macro.lower():
            features['Macro_Context'] = -0.6
        else:
            features['Macro_Context'] = 0.0
        
        # Factor 5: Sentimiento Social (Grok)
        social = decision_inputs.get('social_sentiment', '')
        if 'eufórico' in social.lower() or 'optimista' in social.lower():
            features['Social_Pulse'] = 0.7
        elif 'pesimista' in social.lower() or 'pánico' in social.lower():
            features['Social_Pulse'] = -0.7
        else:
            features['Social_Pulse'] = 0.0
        
        # Factor 6: Análisis Causal
        causal = decision_inputs.get('causal', '')
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', causal)
        if score_match:
            causal_score = float(score_match.group(1))
            features['Causal_Impact'] = min(causal_score, 1.0)
        else:
            features['Causal_Impact'] = 0.1
        
        # Factor 7: Correlación de Mercado
        correlation = decision_inputs.get('correlation', '')
        if 'favorable' in correlation.lower():
            features['Market_Correlation'] = 0.4
        elif 'desfavorable' in correlation.lower():
            features['Market_Correlation'] = -0.4
        else:
            features['Market_Correlation'] = 0.0
        
        # Factor 8: RLHF (Confianza del Operador)
        rlhf = decision_inputs.get('rlhf', '')
        confidence_match = re.search(r'(\d+\.?\d*)', rlhf)
        if confidence_match:
            rlhf_confidence = float(confidence_match.group(1))
            features['Operator_Trust'] = (rlhf_confidence - 0.5) * 2
        else:
            features['Operator_Trust'] = 0.0
        
        # Factor 9: Análisis Multimodal
        multimodal = decision_inputs.get('multimodal', '')
        if 'positiv' in multimodal.lower():
            features['Video_Analysis'] = 0.5
        elif 'negativ' in multimodal.lower():
            features['Video_Analysis'] = -0.5
        else:
            features['Video_Analysis'] = 0.0
        
        return features
    
    def _create_decision_model(self):
        """
        Crea un modelo de decisión mock que simula la lógica de trading.
        
        Returns:
            Función que toma features y retorna probabilidades de decisión
        """
        def decision_function(X):
            predictions = []
            for row in X:
                score = 0
                
                # Pesos para cada factor
                weights = [0.25, 0.20, 0.25, 0.15, 0.10, 0.15, 0.10, 0.20, 0.10]
                
                for i, weight in enumerate(weights):
                    if i < len(row):
                        score += row[i] * weight
                
                # Convertir score a probabilidades
                prob_trade = 1 / (1 + np.exp(-score * 3))  # Sigmoid
                prob_no_trade = 1 - prob_trade
                
                predictions.append([prob_no_trade, prob_trade])
            
            return np.array(predictions)
        
        return decision_function
    
    def _generate_interpretation(self, feature_names: List[str], feature_impacts: List[float], decision: str) -> str:
        """
        Genera una interpretación en lenguaje natural de la explicación.
        
        Args:
            feature_names: Nombres de las características
            feature_impacts: Impactos de cada característica
            decision: Decisión final
            
        Returns:
            Interpretación textual
        """
        # Encontrar los factores más influyentes
        impact_pairs = list(zip(feature_names, feature_impacts))
        impact_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_positive = [pair for pair in impact_pairs if pair[1] > 0][:2]
        top_negative = [pair for pair in impact_pairs if pair[1] < 0][:2]
        
        interpretation = f"Decisión: {decision}\n\n"
        
        if top_positive:
            interpretation += "Factores que FAVORECIERON la operación:\n"
            for name, impact in top_positive:
                interpretation += f"• {name.replace('_', ' ')}: +{impact:.3f}\n"
        
        if top_negative:
            interpretation += "\nFactores que DESFAVORECIERON la operación:\n"
            for name, impact in top_negative:
                interpretation += f"• {name.replace('_', ' ')}: {impact:.3f}\n"
        
        # Determinar el factor dominante
        dominant_factor = impact_pairs[0]
        interpretation += f"\nFactor dominante: {dominant_factor[0].replace('_', ' ')} ({dominant_factor[1]:+.3f})"
        
        return interpretation
    
    def _create_force_plot(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un gráfico de fuerza tipo SHAP usando Plotly.
        
        Args:
            explanation_data: Datos de la explicación
            
        Returns:
            Datos del gráfico en formato JSON
        """
        try:
            feature_names = explanation_data['feature_names']
            feature_impacts = explanation_data['feature_impacts']
            base_value = explanation_data['base_value']
            
            # Separar impactos positivos y negativos
            positive_impacts = [max(0, impact) for impact in feature_impacts]
            negative_impacts = [min(0, impact) for impact in feature_impacts]
            
            # Crear gráfico
            fig = go.Figure()
            
            # Barras positivas
            fig.add_trace(go.Bar(
                y=[name.replace('_', ' ') for name in feature_names],
                x=positive_impacts,
                orientation='h',
                name='Impacto Positivo',
                marker_color='rgba(0, 128, 0, 0.7)',
                text=[f'+{impact:.3f}' if impact > 0 else '' for impact in positive_impacts],
                textposition='outside'
            ))
            
            # Barras negativas
            fig.add_trace(go.Bar(
                y=[name.replace('_', ' ') for name in feature_names],
                x=negative_impacts,
                orientation='h',
                name='Impacto Negativo',
                marker_color='rgba(255, 0, 0, 0.7)',
                text=[f'{impact:.3f}' if impact < 0 else '' for impact in negative_impacts],
                textposition='outside'
            ))
            
            # Configurar layout
            fig.update_layout(
                title='🔍 Explicación XAI - Gráfico de Fuerza',
                xaxis_title='Impacto en la Decisión',
                yaxis_title='Factores de Análisis',
                barmode='relative',
                height=600,
                font=dict(size=12),
                showlegend=True
            )
            
            # Línea base
            fig.add_vline(x=base_value, line_dash="dash", line_color="gray",
                         annotation_text=f"Valor Base: {base_value:.3f}")
            
            return json.loads(pio.to_json(fig))
            
        except Exception as e:
            print(f"Error creando gráfico de fuerza: {e}")
            return {}
    
    def _create_waterfall_chart(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un gráfico de cascada mostrando cómo cada factor contribuye a la decisión.
        
        Args:
            explanation_data: Datos de la explicación
            
        Returns:
            Datos del gráfico en formato JSON
        """
        try:
            feature_names = explanation_data['feature_names']
            feature_impacts = explanation_data['feature_impacts']
            base_value = explanation_data['base_value']
            
            # Preparar datos para cascada
            x_labels = ['Base'] + [name.replace('_', ' ') for name in feature_names] + ['Final']
            y_values = [base_value] + feature_impacts + [sum(feature_impacts) + base_value]
            
            # Crear gráfico de cascada
            fig = go.Figure(go.Waterfall(
                name="Contribución de Factores",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(feature_impacts) + ["total"],
                x=x_labels,
                textposition="outside",
                text=[f"{val:.3f}" for val in y_values],
                y=y_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="🌊 Gráfico de Cascada - Contribución de Factores",
                showlegend=True,
                height=500
            )
            
            return json.loads(pio.to_json(fig))
            
        except Exception as e:
            print(f"Error creando gráfico de cascada: {e}")
            return {}
    
    def _create_feature_importance_chart(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un gráfico de importancia de características.
        
        Args:
            explanation_data: Datos de la explicación
            
        Returns:
            Datos del gráfico en formato JSON
        """
        try:
            feature_names = explanation_data['feature_names']
            feature_impacts = explanation_data['feature_impacts']
            
            # Calcular importancia absoluta
            importance = [abs(impact) for impact in feature_impacts]
            
            # Ordenar por importancia
            sorted_data = sorted(zip(feature_names, importance, feature_impacts), 
                               key=lambda x: x[1], reverse=True)
            
            names, importances, impacts = zip(*sorted_data)
            
            # Crear gráfico de barras
            colors = ['green' if impact > 0 else 'red' for impact in impacts]
            
            fig = go.Figure(go.Bar(
                x=[name.replace('_', ' ') for name in names],
                y=importances,
                marker_color=colors,
                text=[f"{imp:.3f}" for imp in importances],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="📊 Importancia de Factores en la Decisión",
                xaxis_title="Factores",
                yaxis_title="Importancia Absoluta",
                height=400,
                xaxis_tickangle=-45
            )
            
            return json.loads(pio.to_json(fig))
            
        except Exception as e:
            print(f"Error creando gráfico de importancia: {e}")
            return {}
    
    def get_explanation(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una explicación previamente generada.
        
        Args:
            signal_id: ID de la señal
            
        Returns:
            Datos de explicación o None si no existe
        """
        return self.explanations.get(signal_id, None)
    
    def get_all_explanations(self) -> Dict[str, Any]:
        """
        Obtiene todas las explicaciones almacenadas.
        
        Returns:
            Diccionario con todas las explicaciones
        """
        return self.explanations.copy()


# Instancia global del motor de explicaciones
explanation_engine = ExplanationEngine()


def generate_explanation(decision_inputs: Dict[str, Any], final_decision: str, signal_id: str) -> Dict[str, Any]:
    """
    Función de conveniencia para generar explicaciones.
    
    Args:
        decision_inputs: Inputs de la decisión
        final_decision: Decisión final
        signal_id: ID de la señal
        
    Returns:
        Explicación completa
    """
    return explanation_engine.generate_explanation(decision_inputs, final_decision, signal_id)


def get_explanation(signal_id: str) -> Optional[Dict[str, Any]]:
    """
    Función de conveniencia para obtener explicaciones.
    
    Args:
        signal_id: ID de la señal
        
    Returns:
        Explicación o None
    """
    return explanation_engine.get_explanation(signal_id)


# Función de prueba
def test_explanation_engine():
    """
    Función de prueba para el motor de explicaciones.
    """
    print("🧪 Probando Motor de IA Explicable...")
    
    # Datos de prueba
    test_inputs = {
        'trigger': 'RSI en sobreventa (25.4)',
        'cognitive': 'Análisis Cognitivo: Sentimiento positivo detectado',
        'predictive': {'prediction': 'Alcista', 'confidence': 0.75},
        'macro_context': 'Sentimiento: Positivo - Contexto favorable',
        'social_sentiment': 'Pulso Social: Optimista',
        'causal': 'Análisis Causal: Alta causalidad - Score: 0.85',
        'correlation': 'Correlación: Favorable',
        'rlhf': 'Confianza del operador: 0.9',
        'multimodal': 'Análisis Multimodal: Sentimiento positivo'
    }
    
    # Generar explicación
    explanation = generate_explanation(test_inputs, 'COMPRAR', 'test-signal-123')
    
    print(f"✅ Explicación generada para signal_id: test-signal-123")
    print(f"📊 Factores analizados: {len(explanation.get('feature_names', []))}")
    print(f"🎯 Decisión: {explanation.get('decision', 'N/A')}")
    print(f"📝 Interpretación:\n{explanation.get('interpretation', 'N/A')}")
    
    # Recuperar explicación
    retrieved = get_explanation('test-signal-123')
    if retrieved:
        print("✅ Explicación recuperada correctamente del cache")
    else:
        print("❌ Error recuperando explicación del cache")
    
    print("✅ Prueba del Motor de IA Explicable completada!")


if __name__ == "__main__":
    test_explanation_engine()
