
#!/usr/bin/env python3
"""
Multimodal Analyzer Module for DataNexus - Video and Audio Sentiment Analysis
Este módulo procesa contenido multimodal (video, audio, texto) usando Google Gemini
para análisis de sentimiento avanzado que incluye lenguaje corporal y tono de voz.
"""

import os
import asyncio
import tempfile
import aiohttp
import google.generativeai as genai
from pathlib import Path
import yt_dlp
from logger import log_to_dashboard


class MultimodalAnalyzer:
    """
    Analizador multimodal que procesa video, audio y texto usando Google Gemini
    para análisis de sentimiento que incluye elementos no verbales.
    """
    
    def __init__(self, socketio=None):
        """
        Inicializa el analizador multimodal.
        
        Args:
            socketio: Instancia de SocketIO para logs en tiempo real
        """
        self.socketio = socketio
        
        # Configurar Google Gemini API
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')  # Usar Pro para multimodal
                if self.socketio:
                    log_to_dashboard(self.socketio, "🎥 Google Gemini Pro configurado para análisis multimodal")
            except Exception as e:
                if self.socketio:
                    log_to_dashboard(self.socketio, f"❌ Error configurando Gemini Pro: {e}")
                self.gemini_model = None
        else:
            if self.socketio:
                log_to_dashboard(self.socketio, "⚠️ GOOGLE_API_KEY no encontrada. Análisis multimodal no disponible.")
            self.gemini_model = None
    
    async def analyze_video_url(self, video_url: str, context: str = "financial_news") -> str:
        """
        Analiza una URL de video para sentimiento multimodal.
        
        Args:
            video_url: URL del video a analizar (YouTube, news feeds, etc.)
            context: Contexto del análisis ('financial_news', 'earnings_call', 'fed_speech', etc.)
            
        Returns:
            str: Análisis de sentimiento multimodal
        """
        if not self.gemini_model:
            return "Análisis Multimodal: Google Gemini Pro no configurado"
        
        video_path = None
        try:
            if self.socketio:
                log_to_dashboard(self.socketio, f"🎥 Iniciando análisis multimodal de: {video_url[:50]}...")
            
            # 1. Descargar y procesar el video
            video_path = await self._download_video(video_url)
            if not video_path:
                return "Análisis Multimodal: Contenido de video no disponible para análisis"
            
            # 2. Subir video a Gemini
            video_file = await self._upload_to_gemini(video_path)
            if not video_file:
                return "Análisis Multimodal: Error procesando contenido de video"
            
            # 3. Crear prompt multimodal específico
            multimodal_prompt = self._create_multimodal_prompt(context)
            
            # 4. Analizar con Gemini
            analysis_result = await self._analyze_with_gemini(video_file, multimodal_prompt)
            
            if self.socketio:
                log_to_dashboard(self.socketio, f"✅ Análisis multimodal completado exitosamente")
            
            return f"Análisis Multimodal: {analysis_result}" if analysis_result else "Análisis Multimodal: Análisis completado sin resultado específico"
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error crítico en análisis multimodal: {str(e)[:100]}...")
            return f"Análisis Multimodal: No disponible para esta señal - Sistema continuando"
        finally:
            # 5. Limpiar archivos temporales independientemente del resultado
            if video_path:
                self._cleanup_temp_files(video_path)
    
    async def _download_video(self, video_url: str) -> str:
        """
        Descarga video desde URL usando yt-dlp con selección flexible de formatos.
        
        Args:
            video_url: URL del video
            
        Returns:
            str: Ruta del archivo descargado o None si falla
        """
        try:
            # Crear directorio temporal
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "video.%(ext)s")
            
            # Lista de formatos en orden de preferencia (del mejor al más básico)
            format_options = [
                'best[height<=720][duration<=120]',  # Preferido: 720p, máximo 2 min
                'best[height<=480][duration<=120]',  # Alternativo: 480p, máximo 2 min
                'best[height<=360][duration<=120]',  # Básico: 360p, máximo 2 min
                'worst[duration<=120]',              # Último recurso: cualquier calidad, máximo 2 min
                'best[duration<=300]/worst[duration<=300]',  # Emergencia: hasta 5 min
                'best/worst'                         # Desesperado: cualquier video
            ]
            
            # Ejecutar descarga en hilo separado
            loop = asyncio.get_event_loop()
            
            def download_video_with_fallback():
                for i, format_selector in enumerate(format_options):
                    try:
                        ydl_opts = {
                            'format': format_selector,
                            'outtmpl': output_path,
                            'quiet': True,
                            'no_warnings': True,
                            'extract_flat': False,
                            'ignoreerrors': False,
                        }
                        
                        # Buscar archivo de cookies de YouTube
                        cookies_paths = [
                            'cookies.txt',
                            'youtube_cookies.txt',
                            '/home/runner/workspace/cookies.txt',
                            '/home/runner/workspace/youtube_cookies.txt',
                            '/home/runner/cookies.txt',
                            './cookies.txt',
                            './youtube_cookies.txt',
                            'youtube.txt'
                        ]
                        
                        cookies_found = False
                        for cookies_path in cookies_paths:
                            if os.path.exists(cookies_path):
                                ydl_opts['cookiefile'] = cookies_path
                                cookies_found = True
                                if self.socketio:
                                    log_to_dashboard(self.socketio, f"🍪 Usando cookies de: {cookies_path}")
                                break
                        
                        if not cookies_found:
                            # Agregar headers para simular navegador real
                            ydl_opts['http_headers'] = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                'Accept-Language': 'en-US,en;q=0.5',
                                'Accept-Encoding': 'gzip, deflate',
                                'Connection': 'keep-alive'
                            }
                            if self.socketio:
                                log_to_dashboard(self.socketio, "⚠️ No se encontraron cookies. Usando headers de navegador.")
                        
                        # Agregar opciones adicionales para evitar bloqueos
                        ydl_opts.update({
                            'retries': 3,
                            'fragment_retries': 3,
                            'skip_unavailable_fragments': True,
                            'extractor_retries': 3,
                            'file_access_retries': 3,
                            'sleep_interval': 1,
                            'max_sleep_interval': 5
                        })
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([video_url])
                            
                            # Encontrar el archivo descargado
                            for file in os.listdir(temp_dir):
                                if file.startswith("video."):
                                    full_path = os.path.join(temp_dir, file)
                                    # Verificar que el archivo tenga contenido
                                    if os.path.getsize(full_path) > 1024:  # Al menos 1KB
                                        return full_path
                            
                    except Exception as format_error:
                        # Si es el último formato, propagar el error
                        if i == len(format_options) - 1:
                            raise format_error
                        # Si no, continuar con el siguiente formato
                        continue
                
                return None
            
            video_path = await loop.run_in_executor(None, download_video_with_fallback)
            
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
                if self.socketio:
                    log_to_dashboard(self.socketio, f"📥 Video descargado: {os.path.basename(video_path)} ({file_size:.1f}MB)")
                return video_path
            else:
                if self.socketio:
                    log_to_dashboard(self.socketio, f"❌ No se pudo analizar el video [{video_url[:50]}...] - Contenido no disponible")
                return None
                
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ No se pudo analizar el video [{video_url[:50]}...] - Error: {str(e)[:100]}...")
            return None
    
    async def _upload_to_gemini(self, video_path: str):
        """
        Sube el video a Google Gemini para análisis.
        
        Args:
            video_path: Ruta del archivo de video
            
        Returns:
            File object de Gemini o None si falla
        """
        try:
            if self.socketio:
                log_to_dashboard(self.socketio, "☁️ Subiendo video a Google Gemini...")
            
            # Verificar que el archivo existe y tiene contenido
            if not os.path.exists(video_path):
                if self.socketio:
                    log_to_dashboard(self.socketio, "❌ Archivo de video no encontrado para subida")
                return None
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                if self.socketio:
                    log_to_dashboard(self.socketio, "❌ Archivo de video vacío")
                return None
            
            # Ejecutar subida en hilo separado con timeout
            loop = asyncio.get_event_loop()
            
            def upload_file():
                # Intentar diferentes tipos MIME si es necesario
                mime_types = ["video/mp4", "video/webm", "video/x-msvideo", "video/*"]
                
                for mime_type in mime_types:
                    try:
                        return genai.upload_file(path=video_path, mime_type=mime_type)
                    except Exception as mime_error:
                        if mime_type == mime_types[-1]:  # Último intento
                            raise mime_error
                        continue
                return None
            
            # Usar timeout para evitar esperas indefinidas
            video_file = await asyncio.wait_for(
                loop.run_in_executor(None, upload_file),
                timeout=120  # 2 minutos máximo
            )
            
            if video_file:
                if self.socketio:
                    log_to_dashboard(self.socketio, f"✅ Video subido a Gemini: {video_file.name}")
                
                # Implementar bucle de sondeo para esperar estado ACTIVE
                if self.socketio:
                    log_to_dashboard(self.socketio, "⏳ Esperando que Gemini procese el video...")
                
                # Bucle de sondeo de estado
                max_wait_time = 60  # 60 segundos máximo
                poll_interval = 2   # Consultar cada 2 segundos
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    try:
                        # Consultar estado del archivo en hilo separado
                        def check_file_state():
                            return genai.get_file(video_file.name)
                        
                        current_file = await loop.run_in_executor(None, check_file_state)
                        
                        if current_file and hasattr(current_file, 'state'):
                            if current_file.state.name == 'ACTIVE':
                                if self.socketio:
                                    log_to_dashboard(self.socketio, f"✅ Video procesado y listo para análisis")
                                return video_file
                            elif current_file.state.name == 'FAILED':
                                if self.socketio:
                                    log_to_dashboard(self.socketio, "❌ Error: Gemini falló al procesar el video")
                                return None
                            else:
                                if self.socketio:
                                    log_to_dashboard(self.socketio, f"⏳ Estado del video: {current_file.state.name} - Esperando...")
                        
                        # Esperar antes del siguiente sondeo
                        await asyncio.sleep(poll_interval)
                        elapsed_time += poll_interval
                        
                    except Exception as polling_error:
                        if self.socketio:
                            log_to_dashboard(self.socketio, f"⚠️ Error consultando estado: {str(polling_error)[:50]}...")
                        await asyncio.sleep(poll_interval)
                        elapsed_time += poll_interval
                
                # Si llegamos aquí, se agotó el tiempo de espera
                if self.socketio:
                    log_to_dashboard(self.socketio, f"❌ Timeout: Video no se activó en {max_wait_time}s")
                return None
            
            return video_file
            
        except asyncio.TimeoutError:
            if self.socketio:
                log_to_dashboard(self.socketio, "❌ Timeout subiendo video a Gemini")
            return None
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error subiendo a Gemini: {str(e)[:100]}...")
            return None
    
    def _create_multimodal_prompt(self, context: str) -> str:
        """
        Crea prompt específico para análisis multimodal según el contexto.
        
        Args:
            context: Contexto del análisis
            
        Returns:
            str: Prompt optimizado para análisis multimodal
        """
        base_prompt = """
Actúa como un analista financiero experto en comunicación no verbal y análisis de sentimiento multimodal.

Analiza este video de un anuncio económico considerando TODOS estos elementos:

1. **LENGUAJE VERBAL:**
   - Palabras clave financieras y económicas utilizadas
   - Términos técnicos, evasivos o de certeza
   - Declaraciones sobre proyecciones, riesgos o oportunidades
   - Uso de jerga específica del sector

2. **TONO DE VOZ Y PARALENGUAJE:**
   - Velocidad del habla (rápida = nerviosismo/urgencia, lenta = control/gravedad)
   - Pausas significativas antes de datos importantes
   - Cambios en el tono al mencionar cifras o pronósticos
   - Inflexiones que sugieren confianza o duda en las proyecciones

3. **LENGUAJE CORPORAL:**
   - Gestos de manos (abiertos = transparencia, cerrados = defensividad)
   - Postura corporal (erguida = seguridad, encorvada = preocupación)
   - Contacto visual directo con la cámara (confianza vs. evasión)
   - Micro-expresiones faciales durante anuncios críticos

4. **COHERENCIA MULTIMODAL:**
   - ¿El lenguaje corporal respalda las declaraciones optimistas/pesimistas?
   - ¿Hay incongruencias entre lo que dice y cómo lo transmite?
   - ¿Transmite confianza, duda, urgencia o evasión?

5. **INDICADORES ESPECÍFICOS FINANCIEROS:**
   - Reacciones físicas al mencionar números o métricas
   - Énfasis gestual en puntos clave de la estrategia
   - Señales de stress o comfort al discutir resultados
"""
        
        context_specific = {
            'financial_news': "Este es un reporte de noticias financieras. Evalúa si el presentador transmite confianza en la información o si hay signos de incertidumbre sobre la situación económica.",
            
            'earnings_call': "Esta es una llamada de resultados corporativos. Analiza si los ejecutivos transmiten confianza en los números reportados o si hay signos de evasión o preocupación.",
            
            'fed_speech': "Este es un discurso de política monetaria. Evalúa si el funcionario transmite certeza sobre las decisiones futuras o si hay ambigüedad intencional.",
            
            'crypto_news': "Este es contenido relacionado con criptomonedas. Analiza si hay euforia excesiva, miedo, o análisis equilibrado.",
            
            'general': "Analiza el contenido general para determinar el sentimiento dominante."
        }
        
        specific_instruction = context_specific.get(context, context_specific['general'])
        
        final_prompt = f"""
{base_prompt}

**CONTEXTO ESPECÍFICO:**
{specific_instruction}

**RESPUESTA REQUERIDA:**
Proporciona un análisis de sentimiento multimodal en el siguiente formato:

"Sentimiento Multimodal: [Muy Positivo/Positivo/Neutral/Negativo/Muy Negativo]
- Verbal: [Resumen de 1 línea del contenido hablado]
- Vocal: [Resumen de 1 línea del tono y paralenguaje]
- Visual: [Resumen de 1 línea del lenguaje corporal]
- Coherencia: [¿Hay alineación entre todos los canales?]
- Confianza del Análisis: [Alta/Media/Baja]"

Sé preciso y objetivo en tu análisis.
"""
        
        return final_prompt
    
    async def _analyze_with_gemini(self, video_file, prompt: str) -> str:
        """
        Realiza el análisis multimodal usando Gemini.
        
        Args:
            video_file: Archivo de video subido a Gemini
            prompt: Prompt para el análisis
            
        Returns:
            str: Resultado del análisis
        """
        try:
            if self.socketio:
                log_to_dashboard(self.socketio, "🧠 Analizando contenido multimodal con Gemini...")
            
            # Ejecutar análisis en hilo separado
            loop = asyncio.get_event_loop()
            
            def generate_analysis():
                response = self.gemini_model.generate_content([video_file, prompt])
                return response.text if response and hasattr(response, 'text') else None
            
            analysis_result = await loop.run_in_executor(None, generate_analysis)
            
            if analysis_result:
                # Limpiar y formatear resultado
                clean_result = analysis_result.strip()
                if self.socketio:
                    log_to_dashboard(self.socketio, f"✅ Análisis multimodal: {clean_result[:100]}...")
                return clean_result
            else:
                return "Error: No se pudo generar análisis multimodal"
                
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error en análisis Gemini: {e}")
            return f"Error en análisis: {str(e)[:50]}..."
    
    def _cleanup_temp_files(self, video_path: str):
        """
        Limpia archivos temporales.
        
        Args:
            video_path: Ruta del archivo a limpiar
        """
        try:
            if video_path and os.path.exists(video_path):
                # Eliminar archivo
                os.remove(video_path)
                # Eliminar directorio temporal si está vacío
                temp_dir = os.path.dirname(video_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                
                if self.socketio:
                    log_to_dashboard(self.socketio, "🧹 Archivos temporales limpiados")
                    
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"⚠️ Error limpiando archivos: {e}")
    
    async def analyze_financial_video_by_keywords(self, search_keywords: str, max_results: int = 1) -> str:
        """
        Busca y analiza videos financieros por palabras clave.
        
        Args:
            search_keywords: Palabras clave de búsqueda (ej: "Fed Powell speech", "Apple earnings")
            max_results: Número máximo de videos a analizar
            
        Returns:
            str: Análisis multimodal consolidado
        """
        try:
            if self.socketio:
                log_to_dashboard(self.socketio, f"🔍 Buscando videos: '{search_keywords}'")
            
            # Buscar videos usando yt-dlp con múltiples intentos
            search_queries = [
                f"ytsearch{max_results}:{search_keywords} financial news",
                f"ytsearch{max_results}:{search_keywords} news today",
                f"ytsearch{max_results}:{search_keywords}"
            ]
            
            videos = []
            loop = asyncio.get_event_loop()
            
            def search_videos_with_fallback():
                for search_url in search_queries:
                    try:
                        ydl_opts = {
                            'quiet': True,
                            'no_warnings': True,
                            'extract_flat': True,
                            'ignoreerrors': True,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            search_results = ydl.extract_info(search_url, download=False)
                            if search_results and 'entries' in search_results and search_results['entries']:
                                # Filtrar videos válidos
                                valid_videos = [v for v in search_results['entries'] if v and 'id' in v]
                                if valid_videos:
                                    return valid_videos
                    except Exception:
                        continue  # Intentar con la siguiente consulta
                return []
            
            videos = await loop.run_in_executor(None, search_videos_with_fallback)
            
            if not videos:
                if self.socketio:
                    log_to_dashboard(self.socketio, f"🔍 No se encontraron videos para: '{search_keywords}'")
                return "Análisis Multimodal: No se encontraron videos relevantes para análisis"
            
            # Intentar analizar videos hasta que uno funcione
            for i, video_info in enumerate(videos[:3]):  # Máximo 3 intentos
                try:
                    video_url = f"https://www.youtube.com/watch?v={video_info['id']}"
                    video_title = video_info.get('title', 'Video sin título')
                    
                    if self.socketio:
                        log_to_dashboard(self.socketio, f"📺 Analizando video {i+1}: {video_title[:50]}...")
                    
                    # Determinar contexto basado en palabras clave
                    context = self._determine_context(search_keywords, video_title)
                    
                    # Realizar análisis multimodal
                    analysis_result = await self.analyze_video_url(video_url, context)
                    
                    # Si el análisis fue exitoso (no contiene "Error" o "No disponible")
                    if analysis_result and "Error" not in analysis_result and "No disponible" not in analysis_result:
                        return f"{analysis_result} | Video: {video_title[:50]}..."
                    
                except Exception as video_error:
                    if self.socketio:
                        log_to_dashboard(self.socketio, f"⚠️ Video {i+1} falló, probando siguiente...")
                    continue
            
            # Si ningún video pudo ser analizado
            return "Análisis Multimodal: Contenido de video no disponible para esta búsqueda"
            
        except Exception as e:
            if self.socketio:
                log_to_dashboard(self.socketio, f"❌ Error crítico en búsqueda multimodal: {str(e)[:100]}...")
            return "Análisis Multimodal: Sistema de búsqueda temporalmente no disponible"
    
    def _determine_context(self, search_keywords: str, video_title: str) -> str:
        """
        Determina el contexto del análisis basado en palabras clave.
        
        Args:
            search_keywords: Palabras clave de búsqueda
            video_title: Título del video
            
        Returns:
            str: Contexto determinado
        """
        combined_text = f"{search_keywords} {video_title}".lower()
        
        if any(word in combined_text for word in ['fed', 'powell', 'central bank', 'monetary policy', 'interest rate']):
            return 'fed_speech'
        elif any(word in combined_text for word in ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'results']):
            return 'earnings_call'
        elif any(word in combined_text for word in ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi']):
            return 'crypto_news'
        else:
            return 'financial_news'


# Función de prueba
async def test_multimodal_analyzer():
    """
    Función de prueba para el analizador multimodal.
    """
    print("🎥 Probando Multimodal Analyzer...")
    
    analyzer = MultimodalAnalyzer()
    
    # Probar búsqueda y análisis de video
    result = await analyzer.analyze_financial_video_by_keywords("Fed Powell speech latest", max_results=1)
    print(f"✅ Análisis multimodal: {result}")
    
    print("🎥 Prueba del Multimodal Analyzer completada.")


if __name__ == "__main__":
    asyncio.run(test_multimodal_analyzer())
