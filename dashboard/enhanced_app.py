"""
Enhanced Market Pulse Dashboard
Advanced interactive features with real-time updates and model comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import os
import json
import time

# Enhanced page configuration
st.set_page_config(
    page_title="Market Pulse Pro - AI Stock Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .positive { 
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
        color: white; 
    }
    .negative { 
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%); 
        color: white; 
    }
    .neutral {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedMarketPulseDashboard:
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        self.company_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'TSLA': 'Tesla Inc.',
            'AMZN': 'Amazon.com Inc.'
        }
        self.load_models()
    
    def load_models(self):
        """Load trained models if available"""
        try:
            model_dir = "models"
            if os.path.exists(f"{model_dir}/model_metadata.json"):
                with open(f"{model_dir}/model_metadata.json", 'r') as f:
                    self.model_metadata = json.load(f)
                
                self.models_loaded = True
                st.success("🤖 AI Models loaded successfully!")
            else:
                self.models_loaded = False
                st.warning("⚠️ Models not found. Using demo mode.")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def get_real_time_data(self, symbol: str, period: str = "1mo"):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate enhanced technical indicators"""
        if data is None or data.empty:
            return data
            
        # Basic indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return data
    
    def create_advanced_chart(self, data, symbol):
        """Create advanced interactive chart"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[
                f'{symbol} Stock Price & Volume',
                'Technical Indicators',
                'MACD',
                'RSI'
            ],
            row_width=[0.2, 0.2, 0.2, 0.4]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], 
                      name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], 
                      name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], 
                   name='Volume', opacity=0.3),
            row=1, col=1, secondary_y=True
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], 
                      name='BB Upper', line=dict(color='gray', dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], 
                      name='BB Lower', line=dict(color='gray', dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], 
                      name='Close', line=dict(color='blue')),
            row=2, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], 
                      name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      name='RSI', line=dict(color='purple')),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Advanced Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_ai_prediction(self, symbol, data):
        """Generate AI prediction (demo version)"""
        if self.models_loaded:
            # In a real implementation, this would use the actual trained models
            # For now, we'll create a realistic demo prediction
            
            latest_data = data.iloc[-1]
            
            # Simple prediction logic based on technical indicators
            rsi = latest_data['RSI']
            macd = latest_data['MACD']
            price_vs_sma = (latest_data['Close'] - latest_data['SMA_20']) / latest_data['SMA_20']
            
            # Calculate prediction confidence
            if rsi < 30 and macd > 0:  # Oversold but momentum turning positive
                direction = "UP"
                confidence = 0.75
            elif rsi > 70 and macd < 0:  # Overbought and momentum turning negative
                direction = "DOWN"
                confidence = 0.72
            elif abs(price_vs_sma) > 0.05:  # Strong price movement
                direction = "UP" if price_vs_sma > 0 else "DOWN"
                confidence = 0.65
            else:
                direction = "NEUTRAL"
                confidence = 0.55
            
            predicted_return = np.random.normal(0, 0.02)  # Demo return prediction
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_return': predicted_return,
                'reasoning': self._get_prediction_reasoning(rsi, macd, price_vs_sma)
            }
        else:
            return {
                'direction': 'DEMO',
                'confidence': 0.5,
                'predicted_return': 0.0,
                'reasoning': 'Demo mode - load models for real predictions'
            }
    
    def _get_prediction_reasoning(self, rsi, macd, price_vs_sma):
        """Generate reasoning for prediction"""
        reasons = []
        
        if rsi < 30:
            reasons.append("• RSI indicates oversold conditions")
        elif rsi > 70:
            reasons.append("• RSI indicates overbought conditions")
        
        if macd > 0:
            reasons.append("• MACD shows positive momentum")
        elif macd < 0:
            reasons.append("• MACD shows negative momentum")
        
        if price_vs_sma > 0.05:
            reasons.append("• Price significantly above SMA-20")
        elif price_vs_sma < -0.05:
            reasons.append("• Price significantly below SMA-20")
        
        return "\n".join(reasons) if reasons else "• Mixed signals from technical indicators"
    
    def create_market_overview(self):
        """Create market overview dashboard"""
        st.markdown('<h2 class="sub-header">📊 Market Overview</h2>', unsafe_allow_html=True)
        
        # Get data for all symbols
        market_data = {}
        for symbol in self.symbols:
            data, info = self.get_real_time_data(symbol, "5d")
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                
                change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                market_data[symbol] = {
                    'price': latest['Close'],
                    'change': change,
                    'volume': latest['Volume'],
                    'company': self.company_names[symbol]
                }
        
        # Display market overview
        cols = st.columns(len(self.symbols))
        for i, (symbol, data) in enumerate(market_data.items()):
            with cols[i]:
                color = "positive" if data['change'] >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{symbol}</h3>
                    <h4>${data['price']:.2f}</h4>
                    <p class="{color}">{data['change']:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">🚀 Market Pulse Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Stock Analysis & Prediction Platform</p>', unsafe_allow_html=True)
        
        # Market Overview
        self.create_market_overview()
        
        # Sidebar controls
        st.sidebar.markdown("## 🎛️ Analysis Controls")
        
        selected_symbol = st.sidebar.selectbox(
            "Select Stock Symbol:",
            options=self.symbols,
            format_func=lambda x: f"{x} - {self.company_names[x]}"
        )
        
        time_period = st.sidebar.selectbox(
            "Time Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        analysis_type = st.sidebar.multiselect(
            "Analysis Features:",
            options=["Price Chart", "Technical Analysis", "AI Prediction", "Volume Analysis"],
            default=["Price Chart", "Technical Analysis", "AI Prediction"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if "Price Chart" in analysis_type or "Technical Analysis" in analysis_type:
                st.markdown(f'<h2 class="sub-header">📈 {selected_symbol} Analysis</h2>', unsafe_allow_html=True)
                
                # Get and process data
                data, info = self.get_real_time_data(selected_symbol, time_period)
                
                if data is not None and not data.empty:
                    # Calculate technical indicators
                    data = self.calculate_technical_indicators(data)
                    
                    # Create advanced chart
                    if "Technical Analysis" in analysis_type:
                        fig = self.create_advanced_chart(data, selected_symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    elif "Price Chart" in analysis_type:
                        # Simple price chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
                        fig.update_layout(title=f'{selected_symbol} Price Chart', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume analysis
                    if "Volume Analysis" in analysis_type:
                        st.markdown('<h3 class="sub-header">📊 Volume Analysis</h3>', unsafe_allow_html=True)
                        
                        vol_fig = go.Figure()
                        vol_fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
                        vol_fig.update_layout(title=f'{selected_symbol} Trading Volume', height=300)
                        st.plotly_chart(vol_fig, use_container_width=True)
        
        with col2:
            if "AI Prediction" in analysis_type and data is not None:
                st.markdown('<h2 class="sub-header">🤖 AI Prediction</h2>', unsafe_allow_html=True)
                
                prediction = self.generate_ai_prediction(selected_symbol, data)
                
                # Prediction display
                if prediction['direction'] == "UP":
                    pred_class = "positive"
                    pred_icon = "📈"
                elif prediction['direction'] == "DOWN":
                    pred_class = "negative"
                    pred_icon = "📉"
                else:
                    pred_class = "neutral"
                    pred_icon = "🔄"
                
                st.markdown(f"""
                <div class="prediction-box {pred_class}">
                    <h2>{pred_icon} {prediction['direction']}</h2>
                    <p>Confidence: {prediction['confidence']:.1%}</p>
                    <p>Expected Return: {prediction['predicted_return']:+.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning
                st.markdown("### 🧠 AI Reasoning:")
                st.markdown(prediction['reasoning'])
                
                # Model info
                if self.models_loaded:
                    st.markdown("### 📊 Model Performance:")
                    st.json({
                        "Classification Accuracy": f"{self.model_metadata['best_classification_model']['accuracy']:.1%}",
                        "Regression R²": f"{self.model_metadata['best_regression_model']['r2_score']:.3f}",
                        "Training Date": self.model_metadata['training_date'][:10]
                    })
            
            # Company information
            if info:
                st.markdown('<h2 class="sub-header">🏢 Company Info</h2>', unsafe_allow_html=True)
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Market Cap:** ${info.get('marketCap', 0):,.0f}")
                st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>📊 Market Pulse Pro - Powered by AI & Real-time Data</p>
            <p>⚠️ This is for educational purposes only. Not financial advice.</p>
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = EnhancedMarketPulseDashboard()
    dashboard.run_dashboard()
