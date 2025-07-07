"""
Telegram Notifier for DataNexus - Real-time Trading Alerts
Este módulo envía notificaciones de Telegram para alertas de trading en tiempo real.
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️  python-telegram-bot no está disponible. Notificaciones deshabilitadas.")

class TelegramNotifier:
    """
    Notificador de Telegram para alertas de trading.
    """
    
    def __init__(self):
        """
        Inicializa el notificador de Telegram.
        """
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        
        if TELEGRAM_AVAILABLE and self.bot_token:
            self.bot = Bot(token=self.bot_token)
        
        # Configurar logging
        logging.getLogger('telegram').setLevel(logging.WARNING)
    
    def is_configured(self) -> bool:
        """
        Verifica si Telegram está correctamente configurado.
        
        Returns:
            True si está configurado, False en caso contrario
        """
        return (TELEGRAM_AVAILABLE and 
                self.bot_token is not None and 
                self.chat_id is not None and 
                self.bot is not None)
    
    async def send_alert(self, message: str) -> bool:
        """
        Envía una alerta de trading por Telegram.
        
        Args:
            message: Mensaje de alerta a enviar
            
        Returns:
            True si se envió exitosamente, False en caso contrario
        """
        if not self.is_configured():
            print("⚠️  Telegram no configurado. Alerta no enviada.")
            return False
        
        try:
            # Formatear mensaje con timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            formatted_message = f"🚨 *DataNexus Alert*\n\n{message}\n\n⏰ {timestamp}"
            
            # Enviar mensaje
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='Markdown'
            )
            
            print("✅ Alerta de Telegram enviada exitosamente")
            return True
            
        except TelegramError as e:
            print(f"❌ Error enviando alerta de Telegram: {e}")
            return False
        except Exception as e:
            print(f"❌ Error inesperado en Telegram: {e}")
            return False
    
    def send_sync(self, message: str) -> bool:
        """
        Versión síncrona para enviar alertas.
        
        Args:
            message: Mensaje a enviar
            
        Returns:
            True si se envió exitosamente, False en caso contrario
        """
        try:
            return asyncio.run(self.send_alert(message))
        except Exception as e:
            print(f"❌ Error en envío síncrono: {e}")
            return False

# Instancia global del notificador
_telegram_notifier = TelegramNotifier()

def send_telegram_alert(message: str) -> bool:
    """
    Función principal para enviar alertas de Telegram.
    
    Args:
        message: Texto del mensaje a enviar
        
    Returns:
        True si se envió exitosamente, False en caso contrario
    """
    return _telegram_notifier.send_sync(message)

async def send_telegram_alert_async(message: str) -> bool:
    """
    Versión asíncrona para enviar alertas de Telegram.
    
    Args:
        message: Texto del mensaje a enviar
        
    Returns:
        True si se envió exitosamente, False en caso contrario
    """
    return await _telegram_notifier.send_alert(message)

def test_telegram_connection() -> bool:
    """
    Prueba la conexión de Telegram enviando un mensaje de test.
    
    Returns:
        True si la prueba fue exitosa, False en caso contrario
    """
    test_message = "🧪 Test de conexión DataNexus - Sistema funcionando correctamente"
    return send_telegram_alert(test_message)

if __name__ == "__main__":
    # Test del notificador
    print("🧪 Probando notificador de Telegram...")
    success = test_telegram_connection()
    if success:
        print("✅ Telegram configurado correctamente")
    else:
        print("❌ Error en configuración de Telegram")
        print("Verifica TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en Secrets")