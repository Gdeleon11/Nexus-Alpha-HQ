# DataNexus - Financial Data Connector

## Overview

DataNexus is a Python-based financial data aggregation system that provides centralized access to multiple financial APIs. The application serves as a unified interface for cryptocurrency, forex, and market data from various sources including Binance, Finnhub, Alpha Vantage, and Yahoo Finance.

The system is built as a Flask web application with an async-capable backend connector that standardizes data formats across different API providers.

## System Architecture

### Frontend Architecture
- **Technology**: HTML5, CSS3, JavaScript (Vanilla)
- **Framework**: Bootstrap 5.1.3 for responsive UI components
- **Icons**: Font Awesome 6.0.0 for visual elements
- **Structure**: Single-page application with API status monitoring interface

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Async Support**: asyncio with aiohttp for non-blocking API calls
- **Data Processing**: pandas for data manipulation and standardization
- **API Integration**: Multiple specialized clients for different financial data providers

### Core Components
1. **Flask Application** (`app.py`): Web server handling HTTP requests and API endpoints
2. **DataNexus Connector** (`datanexus_connector.py`): Centralized API client manager
3. **Market Scanner** (`market_scanner.py`): Automated trading opportunity scanner using RSI analysis
4. **Deep Analysis Engine** (`deep_analysis_engine.py`): Advanced multi-dimensional analysis orchestrator with four AI components (Gemini, TimeGPT, OpenAI, Claude)
5. **Signal Storage** (`signal_storage.py`): Shared storage for provisional trading signals with verification system
6. **Backtester Engine** (`backtester.py`): Strategy performance simulation and metrics calculation with multiple indicator support
7. **Evolutionary Engine** (`evolutionary_engine.py`): Genetic algorithm optimization for finding optimal trading strategies
8. **Web Interface** (`templates/index.html`): Frontend dashboard for monitoring and testing
9. **Static Assets**: CSS styling and JavaScript functionality

## Key Components

### DataNexus Connector Class
- **Purpose**: Centralized management of all financial API connections
- **Design Pattern**: Singleton-like connector with standardized async methods
- **Supported APIs**:
  - KuCoin (Cryptocurrency data - no API key required)
  - Finnhub (Market news - free tier limitations)
  - Alpha Vantage (Market data)
  - Yahoo Finance (Forex, crypto fallback, and intermarket data)

### API Methods Structure
All data retrieval methods follow a standardized async pattern:
- `get_crypto_data()`: Cryptocurrency price data from Binance
- `get_forex_data()`: Forex data from Finnhub
- `get_market_news()`: Financial news from Finnhub
- `get_intermarket_data()`: Correlation data from Yahoo Finance

### Data Standardization
- **Output Format**: All methods return pandas DataFrames with consistent column structure
- **Required Columns**: timestamp, open, high, low, close, volume, source
- **Timestamp Format**: UTC timezone-aware datetime objects
- **Source Tracking**: Each data point tagged with originating API

## Data Flow

1. **Configuration Loading**: API keys loaded from environment variables at startup
2. **Client Initialization**: Individual API clients initialized based on available credentials
3. **Request Processing**: Flask routes handle HTTP requests and delegate to DataNexus connector
4. **Async Data Retrieval**: Connector methods make concurrent API calls using asyncio
5. **Data Standardization**: Raw API responses converted to standardized pandas DataFrames
6. **Response Delivery**: Processed data returned as JSON through Flask endpoints
7. **Frontend Display**: JavaScript processes API responses and updates the user interface

## External Dependencies

### API Providers
- **Binance**: Cryptocurrency exchange data (requires API key and secret)
- **Finnhub**: Financial market data and news (requires API key)
- **Alpha Vantage**: Market data and indicators (requires API key)
- **Yahoo Finance**: Public market data (no API key required)

### Python Libraries
- **Flask**: Web framework for HTTP server
- **asyncio/aiohttp**: Asynchronous HTTP client operations
- **pandas**: Data manipulation and analysis
- **python-binance**: Binance API client
- **finnhub-python**: Finnhub API client
- **alpha-vantage**: Alpha Vantage API client
- **yfinance**: Yahoo Finance data retrieval

### Frontend Dependencies
- **Bootstrap 5.1.3**: CSS framework via CDN
- **Font Awesome 6.0.0**: Icon library via CDN

## Deployment Strategy

### Environment Configuration
- **API Keys**: Stored as environment variables (Replit Secrets)
- **Security**: No API keys hardcoded in source code
- **Graceful Degradation**: System continues to function with partial API availability

### Error Handling Strategy
- **Robust Exception Management**: Each API call wrapped in try-catch blocks
- **Non-Blocking Failures**: Individual API failures don't crash the entire system
- **User Feedback**: Clear error messages displayed in the web interface
- **Logging**: Error details printed to console for debugging

### Scalability Considerations
- **Async Operations**: Non-blocking API calls prevent server bottlenecks
- **Connection Pooling**: aiohttp manages HTTP connection reuse
- **Rate Limiting**: Built-in respect for API rate limits through client libraries

## Changelog

```
Changelog:
- July 05, 2025. Auditoría y reparación completa de comunicación en tiempo real:
  * Auditoria completa del sistema de logger.py confirmando socketio.emit() funcionando correctamente
  * Reemplazados todos los comandos print() en market_scanner.py por log_to_dashboard() para logs en tiempo real
  * Verificado que deep_analysis_engine.py usa send_signal_to_dashboard() para tarjetas de señal finales
  * Confirmado que app.py usa socketio.run() para habilitar capacidades WebSocket
  * Sistema de comunicación en tiempo real completamente reparado entre backend y frontend
  * Dashboard ahora recibe todos los logs y señales en tiempo real a través de WebSockets
- July 04, 2025. Created new Nexus-Alpha dashboard with professional dark theme:
  * Replaced old HTML/CSS/JS with clean monospace interface optimized for trading
  * Added real-time signal panel displaying trading cards with AI analysis results
  * Created live activity log with system monitoring and auto-scroll functionality
  * Updated signal card format to match frontend expectations (verdict, cognitive, predictive, macro)
  * Temporarily modified RSI threshold to 40 for interface testing (from evolved strategy values)
  * Dashboard now provides professional trading interface with three panels: signals, logs, trade history
- July 04, 2025. Initiated Nexus-Eye browser extension development for price verification:
  * Created nexus-eye-extension folder with manifest.json for Chrome extension
  * Configured extension permissions for tabs, debugger, and storage access
  * Added host permissions for major trading platforms: PocketOption, IQOption, Quotex, Binolla
  * Extension designed as "Nexus-Eye" price verification agent for Nexus-Alpha system
  * Manifest includes service worker, popup interface, and icon configuration
  * Created background.js service worker with WebSocket interception capabilities
  * Implemented automatic broker detection and debugger attachment for price monitoring
  * Added real-time price extraction from WebSocket streams with smart filtering
  * Created popup.html interface for extension control panel with price monitoring display
  * Implemented popup.js with Chrome storage integration and DataNexus API communication
  * Added manual verification button connecting to /api/verify_signal endpoint
  * Created custom SVG icon for professional extension appearance
  * Added comprehensive README.md with installation instructions and usage guide
  * Browser extension now provides complete price verification workflow across trading platforms
- July 04, 2025. Integrated Telegram notifications for real-time trading alerts:
  * Created telegram_notifier.py module with TelegramNotifier class using python-telegram-bot library
  * Added send_telegram_alert() function for instant message delivery to configured chat
  * Integrated Telegram alerts into Deep Analysis Engine for BUY/SELL signals only
  * Created _format_telegram_message() method with professional trading signal formatting
  * Added Flask API endpoint POST /api/test_telegram for connection testing
  * System now sends formatted multi-AI analysis alerts with Signal ID and price information
  * Telegram integration requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables
- July 04, 2025. Created quantum portfolio optimization system (quantum_optimizer.py):
  * Implemented QuantumPortfolioOptimizer class using Qiskit for quantum computing strategies
  * Created QUBO (Quadratic Unconstrained Binary Optimization) problem formulation for strategy selection
  * Added quantum simulation using binary representation and superposition principles
  * Integrated QAOA (Quantum Approximate Optimization Algorithm) with fallback to classical optimization
  * Created Flask API endpoint POST /api/quantum_optimize for web-based quantum optimization
  * System uses quantum utility function combining win_rate, profit_factor, risk, and diversification
  * Quantum optimizer selects optimal strategy portfolios respecting risk constraints and performance metrics
- July 04, 2025. Integrated Grok (xAI) for complete five-AI analysis system:
  * Added Grok client configuration in DeepAnalysisEngine using OpenAI-compatible API
  * Created _run_social_sentiment_analysis method with characteristic Grok personality prompt
  * Enhanced Trading Signal Cards to display social media sentiment analysis
  * Updated OpenAI orchestrator prompt to include all five AI dimensions (Technical + Cognitive + Predictive + Macroeconomic + Social)
  * Modified confluence requirements: now needs 3 of 5 factors for BUY/SELL signals
  * System now provides comprehensive analysis: Gemini (cognitive) + TimeGPT (predictive) + OpenAI (decisions) + Claude (macroeconomic) + Grok (social sentiment)
- July 04, 2025. Enhanced Market Scanner with evolutionary strategy integration:
  * Modified Market Scanner to use evolved strategies instead of fixed RSI rules for opportunity detection
  * Added initialize_evolved_strategies method that evolves optimal strategies for each monitored asset at startup
  * Created _check_evolved_strategy_conditions method supporting RSI, MACD, and SMA evolved strategies
  * Implemented _get_current_indicator_value method for real-time indicator monitoring with evolved parameters
  * Added _calculate_macd_manual method for manual MACD calculations with custom periods
  * System now automatically evolves optimal strategies for BTC-USDT, ETH-USDT, EUR_USD, and GBP_JPY
  * Market Scanner dynamically adapts detection logic based on genetically optimized strategy parameters
- July 04, 2025. Created evolutionary strategy optimization system (evolutionary_engine.py):
  * Implemented EvolutionaryEngine class using genetic algorithms for strategy optimization
  * Added support for evolving RSI, MACD, and SMA strategies with customizable parameters
  * Created fitness function combining profit_factor, win_rate, and drawdown metrics
  * Integrated DEAP library support with fallback to basic genetic algorithm implementation
  * Added Flask API endpoint POST /api/evolve_strategy for web-based evolution
  * System evolves optimal strategies through genetic selection, crossover, and mutation
  * Evolutionary approach finds strategies with superior performance compared to fixed parameters
- July 04, 2025. Created comprehensive backtesting system (backtester.py):
  * Implemented BacktesterEngine class with complete strategy simulation capabilities
  * Added support for multiple trading strategies: RSI, MACD, SMA crossover with customizable parameters
  * Created manual indicator calculations for compatibility when pandas_ta is unavailable
  * Integrated performance metrics calculation: total_trades, win_rate, profit_factor, max_drawdown
  * Added Flask API endpoint POST /api/backtest for web-based strategy testing
  * System provides detailed trade tracking with entry/exit points, P&L calculation, and portfolio curve
  * Backtester validates historical data and simulates trades vela by vela for accurate performance assessment
- July 03, 2025. Integrated Claude (Anthropic) for macroeconomic context analysis:
  * Added anthropic package dependency and Claude API client initialization
  * Created _run_macro_context_analysis method with FMI economist persona
  * Implemented asset-specific macroeconomic context generation (crypto, forex, stocks)
  * Enhanced OpenAI decision-making prompt to include fourth AI factor (macro context)
  * Strengthened confluence requirements: now needs 3 of 4 factors for BUY/SELL signals
  * Updated Trading Signal Cards to display macroeconomic sentiment analysis
  * System now provides comprehensive analysis: Technical + Cognitive + Predictive + Macroeconomic
- July 03, 2025. Added signal verification endpoint with dual process architecture:
  * Configured Flask server and market scanner to run in parallel workflows
  * Created shared signal storage module for provisional trading signals between processes
  * Implemented POST /api/verify_signal endpoint with slippage validation (0.01% threshold)
  * Added GET /api/signals endpoint for signal monitoring and status tracking
  * Modified Deep Analysis Engine to store BUY/SELL signals with real-time market prices
  * System now generates provisional signals that await verification by external "Local_Eye" process
- July 03, 2025. Replaced rule-based decisions with OpenAI GPT-4 intelligent analysis:
  * Integrated OpenAI GPT-4o model for advanced trading decision making
  * Created "Nexus-Decide" AI persona with specialized trading expertise and risk management rules
  * Implemented comprehensive prompt engineering with structured analysis of technical, cognitive, and predictive factors
  * Added intelligent fallback system that uses rule-based logic when OpenAI is unavailable
  * Enhanced error handling for quota limits and API connectivity issues
  * System now provides AI-powered decision making with human-level reasoning capabilities
- July 03, 2025. Added final decision-making system with confluence-based trading rules:
  * Created _make_final_decision method that analyzes signal confluence from all three analysis components
  * Implemented sophisticated trading rules: Buy signals (RSI oversold + bullish prediction + non-negative sentiment), Sell signals (RSI overbought + bearish prediction + non-positive sentiment)
  * Added confidence levels for decision strength (ALTA CONFLUENCIA, CONFLUENCIA MODERADA, CONFLUENCIA TÉCNICA)
  * Enhanced Trading Signal Cards to display final verdict with clear BUY/SELL/NO TRADE recommendations
  * System now provides actionable trading decisions instead of just analysis results
- July 03, 2025. Integrated TimeGPT API for real AI-powered price forecasting:
  * Replaced placeholder predictive analysis with actual TimeGPT forecasting using Nixtla SDK
  * Integrated official Nixtla Python SDK for time series predictions
  * Added proper data formatting for TimeGPT (DataFrame with 'ds', 'y', 'unique_id' columns)
  * Implemented forecast horizon of 5 periods with confidence intervals
  * Added intelligent prediction classification (Alcista/Bajista/Plana) based on price change percentage
  * Enhanced error handling for API rate limits and connection issues
- July 03, 2025. Enhanced Deep Analysis Engine with Google Gemini AI integration:
  * Integrated Google Gemini API for real AI-powered sentiment analysis
  * Replaced placeholder cognitive analysis with live news sentiment evaluation
  * Added automatic topic extraction from trading pairs (BTC-USDT → BTC, OANDA:EUR_USD → EUR)
  * Implemented async content generation with proper error handling
  * Enhanced market news integration with topic-specific search functionality
- July 03, 2025. Enhanced get_market_news function with topic parameter support:
  * Added topic parameter for searching symbol/pair-specific news
  * Implemented company news search and general news filtering
  * Added topic formatting for crypto and forex symbols
  * Updated web interface with topic input field and limit controls
- July 03, 2025. Integrated Deep Analysis Engine with Market Scanner:
  * Created DeepAnalysisEngine class with placeholder analysis methods
  * Modified MarketScanner to call deep analysis when opportunities are detected
  * Added multi-dimensional analysis orchestration (quantitative, cognitive, predictive)
  * Implemented formatted "Trading Signal Cards" for comprehensive opportunity reporting
  * System now provides structured analysis output instead of simple print messages
- July 03, 2025. Added Market Scanner frontend integration:
  * Integrated market scanner into web interface with real-time monitoring
  * Added CSS styling for opportunity cards and scanner logs
  * Implemented JavaScript for automated scanning cycles (10-second intervals)
  * Created scanner status indicators and statistics tracking
  * Frontend now displays live RSI analysis and opportunity detection
- July 03, 2025. Created Market Scanner module (market_scanner.py):
  * Automated trading opportunity scanner using RSI analysis
  * Monitors multiple crypto and forex assets continuously
  * Detects oversold (RSI < 30) and overbought (RSI > 70) conditions
  * Includes custom RSI calculation implementation
  * Fallback system: KuCoin for crypto, yfinance for forex when Finnhub premium not available
- July 03, 2025. Fixed KuCoin API connection issues:
  * Corrected API response format handling (KuCoin returns direct arrays, not wrapped in 'data' object)
  * Updated connection test to use actual market data instead of server timestamp
  * Configured KuCoin API key for enhanced rate limits and access
  * All API connections now show green status (working properly)
- July 03, 2025. Replaced Binance with KuCoin for cryptocurrency data due to geographical restrictions
- July 02, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```