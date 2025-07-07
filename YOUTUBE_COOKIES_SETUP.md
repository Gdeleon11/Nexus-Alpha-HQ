
# Configuración de Cookies de YouTube para Análisis Multimodal

## ¿Por qué necesitamos cookies?

YouTube ha implementado medidas anti-bot que bloquean descargas automatizadas. Para eludir esto, el sistema necesita simular una sesión de navegador real con cookies de autenticación.

## Instrucciones de Configuración

### Paso 1: Exportar Cookies del Navegador

**Para Chrome/Edge:**
1. Instala la extensión "Get cookies.txt LOCALLY" desde la Chrome Web Store
2. Ve a YouTube.com e inicia sesión con tu cuenta
3. Haz clic en la extensión y selecciona "Export cookies.txt"
4. Guarda el archivo como `cookies.txt`

**Para Firefox:**
1. Instala el addon "cookies.txt" 
2. Ve a YouTube.com e inicia sesión
3. Haz clic en el addon y exporta las cookies
4. Guarda como `cookies.txt`

### Paso 2: Subir a Replit

1. En Replit, haz clic en "Upload file" o arrastra el archivo
2. Sube `cookies.txt` al directorio raíz del proyecto
3. Verifica que el archivo esté presente ejecutando: `ls -la cookies.txt`

### Paso 3: Verificar Configuración

El sistema automáticamente detectará el archivo de cookies y lo usará para:
- Descargar videos de YouTube para análisis multimodal
- Evitar bloqueos de bot
- Simular sesión de navegador auténtica

### Ubicaciones de Archivos Soportadas

El sistema buscará cookies en estas ubicaciones:
- `cookies.txt` (recomendado)
- `youtube_cookies.txt`
- `./cookies.txt`
- `/home/runner/workspace/cookies.txt`

### Notas Importantes

⚠️ **Seguridad:** Las cookies contienen información de sesión. No las compartas públicamente.

🔄 **Actualización:** Las cookies expiran. Si el sistema falla, exporta nuevas cookies.

✅ **Verificación:** El sistema mostrará "🍪 Usando cookies de: [ruta]" cuando funcione correctamente.

### Resolución de Problemas

Si el análisis multimodal sigue fallando:
1. Verifica que el archivo de cookies esté presente
2. Asegúrate de que el archivo no esté vacío
3. Reexporta cookies desde un navegador con sesión activa de YouTube
4. Verifica que YouTube funcione normalmente en tu navegador

### Formato del Archivo

El archivo debe estar en formato Netscape cookies.txt:
```
# Netscape HTTP Cookie File
.youtube.com	TRUE	/	FALSE	1234567890	cookie_name	cookie_value
```

Una vez configurado, el sistema podrá analizar contenido multimedia de YouTube para análisis de sentimiento financiero.
