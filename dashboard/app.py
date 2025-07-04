"""
Market Pulse - Interactive Dashboard

This Streamlit application provides an interactive dashboard for the
Market Pulse stock sentiment analysis and price prediction system.
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

# Page configuration
st.set_page_config(
    page_title="Market Pulse - AI Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📈 Market Pulse: AI-Driven Stock Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
**Market Pulse** combines advanced sentiment analysis with technical indicators to predict stock price movements.
This interactive dashboard demonstrates the power of AI in financial market analysis.
""")

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Stock selection
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
selected_symbol = st.sidebar.selectbox(
    "📊 Select Stock Symbol",
    SYMBOLS,
    index=0,
    help="Choose a stock symbol for analysis"
)

# Time period selection
time_periods = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}
selected_period = st.sidebar.selectbox(
    "📅 Time Period",
    list(time_periods.keys()),
    index=2,
    help="Select the historical data period"
)

# Analysis type
analysis_type = st.sidebar.radio(
    "🔍 Analysis Type",
    ["Overview", "Technical Analysis", "Sentiment Analysis", "Predictions"],
    help="Choose the type of analysis to display"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 About Market Pulse
- **Data Sources**: Yahoo Finance, News APIs
- **AI Models**: Random Forest, XGBoost
- **Sentiment Analysis**: VADER, TextBlob
- **Technical Indicators**: 15+ indicators
""")

@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['SMA_20']
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

def generate_sample_sentiment():
    """Generate sample sentiment data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    sentiment_data = []
    for date in dates:
        sentiment_data.append({
            'date': date,
            'daily_sentiment': np.random.normal(0, 0.3),
            'news_volume': np.random.randint(1, 15),
            'sentiment_category': np.random.choice(['positive', 'neutral', 'negative'], p=[0.3, 0.4, 0.3])
        })
    
    return pd.DataFrame(sentiment_data)

def create_price_chart(data, symbol):
    """Create interactive price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Stock Price', 'Volume', 'Technical Indicators'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{symbol} Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            name='SMA 50',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Bollinger Bands',
            fillcolor='rgba(0,100,80,0.2)'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """Create sentiment analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Sentiment Trend', 'Sentiment Distribution', 
                       'News Volume Over Time', 'Sentiment Category Breakdown'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Sentiment trend
    fig.add_trace(
        go.Scatter(
            x=sentiment_data['date'],
            y=sentiment_data['daily_sentiment'],
            mode='lines+markers',
            name='Daily Sentiment',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Sentiment distribution
    fig.add_trace(
        go.Histogram(
            x=sentiment_data['daily_sentiment'],
            nbinsx=20,
            name='Sentiment Distribution',
            marker_color='skyblue'
        ),
        row=1, col=2
    )
    
    # News volume
    fig.add_trace(
        go.Bar(
            x=sentiment_data['date'],
            y=sentiment_data['news_volume'],
            name='News Volume',
            marker_color='lightgreen'
        ),
        row=2, col=1
    )
    
    # Sentiment categories
    category_counts = sentiment_data['sentiment_category'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            name='Sentiment Categories'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def make_prediction(current_price, rsi, sentiment):
    """Simple prediction logic (demo purposes)"""
    # This is a simplified prediction for demo purposes
    # In reality, you would use the trained ML models
    
    score = 0
    
    # Technical analysis component
    if rsi < 30:
        score += 0.3  # Oversold, likely to go up
    elif rsi > 70:
        score -= 0.3  # Overbought, likely to go down
    
    # Sentiment component
    score += sentiment * 0.5
    
    # Add some randomness to simulate model uncertainty
    score += np.random.normal(0, 0.1)
    
    # Convert to probability
    probability = 1 / (1 + np.exp(-score * 2))  # Sigmoid function
    
    direction = "UP" if probability > 0.5 else "DOWN"
    confidence = abs(probability - 0.5) * 2
    
    if confidence > 0.7:
        confidence_level = "HIGH"
    elif confidence > 0.5:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    predicted_return = score * 0.02  # Convert to percentage
    
    return {
        'direction': direction,
        'probability': probability,
        'confidence_level': confidence_level,
        'predicted_return': predicted_return
    }

# Main application logic
def main():
    # Load data
    with st.spinner(f"📊 Loading data for {selected_symbol}..."):
        stock_data = load_stock_data(selected_symbol, time_periods[selected_period])
        sentiment_data = generate_sample_sentiment()
    
    if stock_data is None:
        st.error("❌ Failed to load stock data. Please try again.")
        return
    
    # Overview Section
    if analysis_type == "Overview":
        st.header("📊 Market Overview")
        
        # Key metrics
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"{selected_symbol} Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Volume",
                value=f"{volume:,.0f}",
                delta=f"{((volume / avg_volume - 1) * 100):.1f}% vs avg"
            )
        
        with col3:
            high_52w = stock_data['High'].max()
            low_52w = stock_data['Low'].min()
            st.metric(
                label="52W High",
                value=f"${high_52w:.2f}"
            )
            st.metric(
                label="52W Low", 
                value=f"${low_52w:.2f}"
            )
        
        with col4:
            rsi = stock_data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="RSI",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
        
        # Price chart
        st.plotly_chart(create_price_chart(stock_data, selected_symbol), use_container_width=True)
    
    # Technical Analysis
    elif analysis_type == "Technical Analysis":
        st.header("🔧 Technical Analysis")
        
        # Technical indicators summary
        latest_data = stock_data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Moving Averages")
            sma_20 = latest_data['SMA_20']
            sma_50 = latest_data['SMA_50']
            current_price = latest_data['Close']
            
            ma_signal = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Mixed"
            st.write(f"**Signal**: {ma_signal}")
            st.write(f"Price vs SMA 20: {((current_price / sma_20 - 1) * 100):.2f}%")
            st.write(f"Price vs SMA 50: {((current_price / sma_50 - 1) * 100):.2f}%")
        
        with col2:
            st.subheader("⚡ Momentum Indicators")
            rsi = latest_data['RSI']
            macd = latest_data['MACD']
            macd_signal = latest_data['MACD_Signal']
            
            rsi_interpretation = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            macd_interpretation = "Bullish" if macd > macd_signal else "Bearish"
            
            st.write(f"**RSI**: {rsi:.1f} ({rsi_interpretation})")
            st.write(f"**MACD**: {macd_interpretation} crossover")
            st.write(f"MACD Line: {macd:.4f}")
            st.write(f"Signal Line: {macd_signal:.4f}")
        
        # Detailed technical chart
        st.plotly_chart(create_price_chart(stock_data, selected_symbol), use_container_width=True)
    
    # Sentiment Analysis
    elif analysis_type == "Sentiment Analysis":
        st.header("🧠 Sentiment Analysis")
        
        # Sentiment metrics
        avg_sentiment = sentiment_data['daily_sentiment'].mean()
        sentiment_volatility = sentiment_data['daily_sentiment'].std()
        total_news = sentiment_data['news_volume'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric(
                label="Average Sentiment",
                value=f"{avg_sentiment:.3f}",
                delta=sentiment_label
            )
        
        with col2:
            st.metric(
                label="Sentiment Volatility",
                value=f"{sentiment_volatility:.3f}"
            )
        
        with col3:
            st.metric(
                label="Total News Articles",
                value=f"{total_news:,.0f}"
            )
        
        # Sentiment charts
        st.plotly_chart(create_sentiment_chart(sentiment_data), use_container_width=True)
        
        # Recent news sentiment
        st.subheader("📰 Recent News Sentiment")
        recent_sentiment = sentiment_data.tail(10)
        
        for _, row in recent_sentiment.iterrows():
            sentiment_class = "positive" if row['daily_sentiment'] > 0.1 else "negative" if row['daily_sentiment'] < -0.1 else "neutral"
            
            st.markdown(f"""
            <div class="prediction-box {sentiment_class}">
                {row['date'].strftime('%Y-%m-%d')}: Sentiment {row['daily_sentiment']:.3f} 
                ({row['sentiment_category'].title()}) - {row['news_volume']} articles
            </div>
            """, unsafe_allow_html=True)
    
    # Predictions
    elif analysis_type == "Predictions":
        st.header("🔮 AI Predictions")
        
        # Get current data for prediction
        current_price = stock_data['Close'].iloc[-1]
        current_rsi = stock_data['RSI'].iloc[-1]
        current_sentiment = sentiment_data['daily_sentiment'].iloc[-1]
        
        # Make prediction
        prediction = make_prediction(current_price, current_rsi, current_sentiment)
        
        # Display prediction
        st.subheader(f"🎯 Prediction for {selected_symbol}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_class = "positive" if prediction['direction'] == "UP" else "negative"
            st.markdown(f"""
            <div class="prediction-box {direction_class}">
                <h3>Price Direction</h3>
                <h2>{prediction['direction']}</h2>
                <p>Probability: {prediction['probability']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_class = "positive" if prediction['confidence_level'] == "HIGH" else "neutral" if prediction['confidence_level'] == "MEDIUM" else "negative"
            st.markdown(f"""
            <div class="prediction-box {confidence_class}">
                <h3>Confidence Level</h3>
                <h2>{prediction['confidence_level']}</h2>
                <p>Model Certainty</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            return_class = "positive" if prediction['predicted_return'] > 0 else "negative"
            st.markdown(f"""
            <div class="prediction-box {return_class}">
                <h3>Expected Return</h3>
                <h2>{prediction['predicted_return']:.2%}</h2>
                <p>Next 1 Day</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance (mock data)
        st.subheader("🔍 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['RSI', 'Daily Sentiment', 'MACD', 'News Volume', 'Price Momentum', 'Volume Ratio'],
            'Importance': [0.25, 0.22, 0.18, 0.15, 0.12, 0.08]
        })
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Model Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Disclaimer
        st.warning("""
        ⚠️ **Disclaimer**: These predictions are for educational purposes only and should not be used as financial advice. 
        Stock market investments carry inherent risks, and past performance does not guarantee future results.
        Always consult with a qualified financial advisor before making investment decisions.
        """)

# Run the application
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📈 <strong>Market Pulse</strong> - AI-Driven Stock Analysis | Built with ❤️ using Streamlit</p>
    <p>🔗 <a href="https://github.com/yourusername/market-pulse-ai" target="_blank">GitHub Repository</a> | 
       📧 <a href="mailto:your.email@example.com">Contact</a></p>
</div>
""", unsafe_allow_html=True)
