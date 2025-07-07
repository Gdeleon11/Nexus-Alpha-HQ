
import logging
from datetime import datetime
import os

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_to_dashboard(socketio_instance, message):
    """
    Envía un mensaje de log al dashboard via SocketIO
    
    Args:
        socketio_instance: Instancia de SocketIO para emitir el mensaje
        message: Mensaje a enviar
    """
    try:
        if socketio_instance:
            formatted_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            socketio_instance.emit('new_log', {'data': formatted_message})
            print(formatted_message)  # También imprimir en consola
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    except Exception as e:
        print(f"Error enviando log al dashboard: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def send_signal_to_dashboard(socketio_instance, signal_data):
    """
    Envía una señal de trading al dashboard via SocketIO
    
    Args:
        socketio_instance: Instancia de SocketIO para emitir la señal
        signal_data: Datos de la señal a enviar
    """
    try:
        if socketio_instance:
            # Asegurar que la señal tenga timestamp
            if 'timestamp' not in signal_data:
                signal_data['timestamp'] = datetime.now().strftime('%H:%M:%S')
            
            # Asegurar que la señal tenga expiration si no existe
            if 'expiration' not in signal_data:
                signal_data['expiration'] = 'No especificada'
            
            socketio_instance.emit('new_signal', signal_data)
            print(f"--- TARJETA DE SEÑAL ENVIADA A SOCKETIO ---")
        else:
            print(f"No se pudo enviar señal - SocketIO no disponible: {signal_data}")
    except Exception as e:
        print(f"Error enviando señal al dashboard: {e}")

def create_logger(name):
    """
    Crear un logger específico para un módulo
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Crear handler de archivo si no existe
    if not logger.handlers:
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Handler de archivo
        file_handler = logging.FileHandler(f'logs/{name}.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Agregar handler al logger
        logger.addHandler(file_handler)
    
    return logger

# Logger global para el sistema
system_logger = create_logger('nexus_alpha_system')

def log_system_event(event_type, message, level='INFO'):
    """
    Registrar un evento del sistema
    
    Args:
        event_type: Tipo de evento (SCANNER, ANALYSIS, ERROR, etc.)
        message: Mensaje del evento
        level: Nivel de logging (INFO, WARNING, ERROR)
    """
    formatted_message = f"[{event_type}] {message}"
    
    if level == 'ERROR':
        system_logger.error(formatted_message)
    elif level == 'WARNING':
        system_logger.warning(formatted_message)
    else:
        system_logger.info(formatted_message)

def log_trading_signal(signal_data):
    """
    Registrar una señal de trading en logs
    
    Args:
        signal_data: Datos de la señal de trading
    """
    signal_logger = create_logger('trading_signals')
    signal_logger.info(f"SIGNAL_GENERATED: {signal_data}")

def log_analysis_result(pair, timeframe, analysis_result):
    """
    Registrar resultado de análisis
    
    Args:
        pair: Par analizado
        timeframe: Timeframe del análisis
        analysis_result: Resultado del análisis
    """
    analysis_logger = create_logger('analysis_results')
    analysis_logger.info(f"ANALYSIS: {pair} ({timeframe}) - {analysis_result}")
