# 🚀 Market Pulse - AI-Driven Stock Sentiment Analysis & Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/Zeesejo/-Market-Pulse---AI-Driven-Stock-Sentiment-Analysis.svg)](https://github.com/Zeesejo/-Market-Pulse---AI-Driven-Stock-Sentiment-Analysis/releases/)

> An advanced AI-powered system that combines Natural Language Processing with financial technical analysis to predict stock price movements through sentiment analysis of financial news.

## 🎯 Live Demo

🌐 **[Try the Interactive Dashboard](http://localhost:8501)** (Run locally)

## 📊 Project Overview

Market Pulse is a comprehensive data science project that demonstrates the integration of:
- **Real-time Financial Data** collection and processing
- **Advanced NLP** for sentiment analysis of financial news
- **Machine Learning** models for price prediction
- **Interactive Dashboards** for real-time analysis
- **Professional Documentation** following industry standards

### 🎬 Quick Preview
![Market Pulse Dashboard Preview](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Market+Pulse+Dashboard)

## ✨ Key Features

### 📈 **Real-Time Data Integration**
- Live stock price data from Yahoo Finance API
- Technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA)
- Volume analysis and volatility metrics
- Support for major tech stocks (AAPL, MSFT, GOOGL, TSLA, AMZN)

### 🧠 **Advanced Sentiment Analysis**
- Multi-method approach (VADER, TextBlob)
- Financial news processing and aggregation
- Sentiment-based feature engineering
- Real-time news impact assessment

### 🤖 **Machine Learning Pipeline**
- **Classification Models**: Price direction prediction (Up/Down)
- **Regression Models**: Future return estimation
- **Feature Engineering**: 49+ technical and sentiment features
- **Model Selection**: XGBoost, Random Forest, Logistic Regression

### 📊 **Interactive Dashboard**
- Professional Streamlit interface with real-time updates
- Advanced charting with Plotly (candlestick, technical indicators)
- AI prediction display with confidence levels
- Market overview for multiple stocks

### 📋 **Production Ready**
- Comprehensive documentation and research journal
- Deployment scripts and model persistence
- Error handling and logging
- Extensible architecture for real APIs

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zeesejo/-Market-Pulse---AI-Driven-Stock-Sentiment-Analysis.git
   cd -Market-Pulse---AI-Driven-Stock-Sentiment-Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis notebook**
   ```bash
   jupyter notebook notebooks/market_pulse_analysis.ipynb
   ```

5. **Launch the dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

### 🎯 Usage Examples

#### Basic Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```

#### Enhanced Dashboard with Advanced Features
```bash
streamlit run dashboard/enhanced_app.py --server.port 8502
```

#### Real-time Prediction
```python
from deployment.predictor import MarketPulsePredictior

predictor = MarketPulsePredictior()
prediction = predictor.predict_comprehensive(features_dict)
print(f"Direction: {prediction['price_direction']} ({prediction['confidence']:.1%} confidence)")
```

## 📁 Project Structure

```
market-pulse/
├── 📂 data/                     # Data storage
├── 📂 src/                      # Core modules
│   ├── data_collection.py       # Stock & news data collection
│   ├── sentiment_analysis.py    # NLP sentiment processing
│   ├── ml_models.py             # Machine learning pipeline
│   ├── real_news_collector.py   # Real-time news APIs
│   └── advanced_ml.py           # Enhanced ML techniques
├── 📂 models/                   # Trained models
│   ├── best_classification_model.joblib
│   ├── best_regression_model.joblib
│   └── model_metadata.json
├── 📂 notebooks/                # Analysis notebooks
│   └── market_pulse_analysis.ipynb
├── 📂 dashboard/                # Interactive dashboards
│   ├── app.py                   # Basic Streamlit app
│   └── enhanced_app.py          # Advanced dashboard
├── 📂 deployment/               # Production deployment
│   └── predictor.py             # Model inference
├── 📂 docs/                     # Documentation
│   ├── research_journal.md      # Academic documentation
│   └── project_summary.md       # Executive summary
└── 📋 Configuration files
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

## 📊 Model Performance

### Current Results (v1.0)
| Model Type | Algorithm | Performance | Notes |
|------------|-----------|-------------|-------|
| **Classification** | XGBoost | 49.6% accuracy | Price direction prediction |
| **Regression** | Random Forest | R² = -0.169 | Future return estimation |
| **Features** | Technical + Sentiment | 49 features | Real-time processing |

### 🎯 Enhancement Roadmap
- **Real Data Integration**: 55-65% expected accuracy
- **Ensemble Methods**: 60-70% expected accuracy  
- **Deep Learning**: 70-80% expected accuracy
- **Production APIs**: Real-time deployment ready

## 🛠️ Technical Stack

### **Core Technologies**
- **Python 3.9+**: Primary language
- **pandas & NumPy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning framework
- **XGBoost**: Gradient boosting for predictions

### **Data & APIs**
- **yfinance**: Real-time stock data
- **NewsAPI & Finnhub**: Financial news feeds (optional)
- **VADER & TextBlob**: Sentiment analysis

### **Visualization & UI**
- **Streamlit**: Interactive web dashboard
- **Plotly**: Advanced financial charts
- **Matplotlib & Seaborn**: Statistical visualizations

### **Development & Deployment**
- **Jupyter**: Notebook-based development
- **Git**: Version control
- **joblib**: Model persistence

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file for API keys:
```env
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
```

### Streamlit Configuration
Customize dashboard in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## 📚 Documentation

- **[📖 Research Journal](docs/research_journal.md)**: Academic-style methodology and results
- **[📋 Project Summary](docs/project_summary.md)**: Executive summary and insights  
- **[🚀 Enhancement Guide](ENHANCEMENT_GUIDE.md)**: Advanced features and scaling options
- **[🔧 Troubleshooting](TROUBLESHOOTING.md)**: Common issues and solutions
- **[⚙️ GitHub Setup](GITHUB_SETUP.md)**: Repository deployment guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🎯 Use Cases

### **Academic & Research**
- Data science portfolio project
- Financial ML research
- NLP sentiment analysis studies
- Time series forecasting research

### **Professional Development**
- MLOps pipeline demonstration
- Real-time dashboard development
- API integration showcase
- Production deployment practices

### **Business Applications**
- Investment decision support
- Risk management tools
- Market sentiment monitoring
- Automated trading signals

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run dashboard/app.py
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py"]
```

### **Cloud Platforms**
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app deployment
- **AWS/Azure/GCP**: Scalable cloud infrastructure

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions and analysis provided should **not be considered as financial advice**. Always consult with qualified financial advisors before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance API** for real-time stock data
- **VADER Sentiment** for financial text analysis
- **Streamlit** for the amazing dashboard framework
- **Plotly** for interactive financial charts
- **scikit-learn & XGBoost** for machine learning capabilities

---

## 📞 Contact

**Project Repository**: [https://github.com/Zeesejo/-Market-Pulse---AI-Driven-Stock-Sentiment-Analysis](https://github.com/Zeesejo/-Market-Pulse---AI-Driven-Stock-Sentiment-Analysis)

⭐ **Star this repository if you found it helpful!**

---

*Built with ❤️ for the data science and financial technology community*
