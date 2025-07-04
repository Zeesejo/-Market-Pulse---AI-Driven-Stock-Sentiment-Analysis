# 🔧 Market Pulse - Troubleshooting Guide

## Common Issues & Solutions

### 1. **Plotly Subplot Secondary Y-axis Error** ✅ FIXED
**Error:** `ValueError: Subplot with type '{subplot_type}' was not created with secondary_y spec property`

**Solution:** Updated `dashboard/enhanced_app.py` with proper subplot specs:
```python
specs=[[{"secondary_y": True}],
       [{"secondary_y": False}],
       [{"secondary_y": False}],
       [{"secondary_y": False}]]
```

### 2. **Port Already in Use**
**Error:** `Port 8501 is already in use`

**Solutions:**
```bash
# Option A: Use different port
streamlit run dashboard/app.py --server.port 8502

# Option B: Kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

### 3. **Module Import Errors**
**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Activate virtual environment
cd "e:\Projects\P04"
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install missing packages
pip install -r requirements.txt
```

### 4. **Model Files Not Found**
**Error:** `FileNotFoundError: models/xxx.joblib`

**Solution:**
Run the main notebook first to generate models:
```bash
jupyter notebook notebooks/market_pulse_analysis.ipynb
# Execute all cells to train and save models
```

### 5. **API Key Issues** (Real Data Integration)
**Error:** `401 Unauthorized` or `403 Forbidden`

**Solution:**
1. Get free API keys:
   - [NewsAPI](https://newsapi.org/) - 100 requests/day
   - [Finnhub](https://finnhub.io/) - 60 calls/minute

2. Update `src/real_news_collector.py`:
```python
api_keys = {
    'newsapi': 'YOUR_ACTUAL_API_KEY',
    'finnhub': 'YOUR_ACTUAL_API_KEY'
}
```

### 6. **Yahoo Finance Data Issues**
**Error:** `KeyError` or empty data from yfinance

**Solutions:**
```python
# Check symbol validity
import yfinance as yf
stock = yf.Ticker("AAPL")
info = stock.info
print(info.get('longName', 'Symbol not found'))

# Use different period if data is missing
data = stock.history(period="1mo")  # Instead of "1y"
```

### 7. **Memory Issues with Large Datasets**
**Error:** `MemoryError` or slow performance

**Solutions:**
```python
# Reduce data size
data = data.tail(1000)  # Last 1000 rows only

# Use data types optimization
data['Close'] = data['Close'].astype('float32')

# Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

### 8. **Unicode Encoding Issues**
**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:**
Always use UTF-8 encoding for file operations:
```python
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write(text_with_emojis)
```

### 9. **Jupyter Notebook Kernel Issues**
**Error:** Kernel not starting or crashing

**Solutions:**
```bash
# Restart kernel in notebook
# Kernel → Restart & Clear Output

# Or from command line:
jupyter kernelspec list
jupyter kernelspec uninstall old_kernel
jupyter kernelspec install-self --user

# Reinstall jupyter
pip uninstall jupyter notebook
pip install jupyter notebook
```

### 10. **Streamlit Configuration Issues**
**Error:** Dashboard not loading properly

**Solution:**
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
port = 8501
```

## 🚀 Performance Optimization Tips

### 1. **Caching**
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.joblib')
```

### 2. **Async Data Loading**
```python
import asyncio
import aiohttp

async def fetch_multiple_stocks(symbols):
    tasks = [fetch_stock_data(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

### 3. **Data Preprocessing**
```python
# Use vectorized operations
data['returns'] = data['Close'].pct_change()

# Avoid loops, use pandas methods
data['sma'] = data['Close'].rolling(20).mean()
```

## 📞 Getting Help

### Quick Fixes:
1. **Restart everything**: Kernel → Dashboard → Python environment
2. **Check file paths**: All paths should be absolute or relative to project root
3. **Update packages**: `pip install --upgrade -r requirements.txt`
4. **Clear cache**: Delete `.streamlit/` folder and restart

### Debug Mode:
```python
# Add to dashboard
st.write("Debug info:", locals())
st.write("Data shape:", data.shape)
st.write("Columns:", data.columns.tolist())
```

### Log Analysis:
```bash
# Check Streamlit logs
streamlit run app.py --logger.level debug

# Python logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ✅ Current Status
- **Main Dashboard**: ✅ Working (port 8501)
- **Enhanced Dashboard**: ✅ Fixed and working (port 8503)
- **Models**: ✅ Trained and saved
- **Notebook**: ✅ All cells executed successfully
- **Documentation**: ✅ Complete

Your Market Pulse project is fully functional! 🎉
