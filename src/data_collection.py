"""
Market Pulse - Data Collection Module

This module handles the collection of financial news and stock price data
for sentiment analysis and price prediction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple
import os
from dataclasses import dataclass
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data structure for news articles"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    symbol: str

@dataclass
class StockData:
    """Data structure for stock price data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float

class StockDataCollector:
    """Collects historical and real-time stock data"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with stock price data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str] = None, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical data for multiple stock symbols
        
        Args:
            symbols: List of stock symbols
            period: Time period
        
        Returns:
            Combined DataFrame with all stock data
        """
        if symbols is None:
            symbols = self.symbols
        
        all_data = []
        
        for symbol in symbols:
            data = self.get_historical_data(symbol, period)
            if not data.empty:
                all_data.append(data)
            time.sleep(0.1)  # Rate limiting
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully fetched data for {len(symbols)} symbols")
            return combined_data
        else:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data
        
        Args:
            data: DataFrame with stock price data
        
        Returns:
            DataFrame with technical indicators added
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df.groupby('Symbol')['Close'].rolling(window=10).mean().reset_index(0, drop=True)
        df['SMA_20'] = df.groupby('Symbol')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
        df['SMA_50'] = df.groupby('Symbol')['Close'].rolling(window=50).mean().reset_index(0, drop=True)
        
        # Exponential Moving Averages
        df['EMA_12'] = df.groupby('Symbol')['Close'].ewm(span=12).mean().reset_index(0, drop=True)
        df['EMA_26'] = df.groupby('Symbol')['Close'].ewm(span=26).mean().reset_index(0, drop=True)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df.groupby('Symbol')['MACD'].ewm(span=9).mean().reset_index(0, drop=True)
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI'] = df.groupby('Symbol')['Close'].apply(calculate_rsi).reset_index(0, drop=True)
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        bb_std = df.groupby('Symbol')['Close'].rolling(window=20).std().reset_index(0, drop=True)
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price changes
        df['Price_Change'] = df.groupby('Symbol')['Close'].pct_change()
        df['Price_Change_1d'] = df.groupby('Symbol')['Close'].pct_change(periods=1)
        df['Price_Change_5d'] = df.groupby('Symbol')['Close'].pct_change(periods=5)
        
        # Volume indicators
        df['Volume_SMA_10'] = df.groupby('Symbol')['Volume'].rolling(window=10).mean().reset_index(0, drop=True)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        logger.info("Technical indicators calculated successfully")
        return df

class NewsDataCollector:
    """Collects financial news data from various sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
    def search_news(self, query: str, days_back: int = 7, language: str = 'en') -> List[Dict]:
        """
        Search for news articles using NewsAPI
        
        Args:
            query: Search query
            days_back: Number of days to look back
            language: Language code
        
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No News API key provided. Using sample data.")
            return self._get_sample_news_data(query)
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'language': language,
                'sortBy': 'publishedAt',
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news for query '{query}': {str(e)}")
            return self._get_sample_news_data(query)
    
    def get_stock_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """
        Get news articles for a specific stock symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            List of NewsArticle objects
        """
        # Search terms for the stock
        company_names = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google Alphabet',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon'
        }
        
        query = f"{symbol} OR {company_names.get(symbol, symbol)}"
        articles_data = self.search_news(query, days_back)
        
        news_articles = []
        for article in articles_data:
            try:
                news_article = NewsArticle(
                    title=article.get('title', ''),
                    content=article.get('content', '') or article.get('description', ''),
                    source=article.get('source', {}).get('name', ''),
                    published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                    url=article.get('url', ''),
                    symbol=symbol
                )
                news_articles.append(news_article)
            except Exception as e:
                logger.warning(f"Error processing article: {str(e)}")
                continue
        
        return news_articles
    
    def _get_sample_news_data(self, query: str) -> List[Dict]:
        """Generate sample news data when API is not available"""
        sample_articles = [
            {
                'title': f'{query} Reports Strong Q4 Earnings',
                'content': f'{query} announced better-than-expected quarterly results with revenue growth of 15%.',
                'source': {'name': 'Financial Times'},
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat() + 'Z',
                'url': 'https://example.com/article1'
            },
            {
                'title': f'{query} Announces New Product Launch',
                'content': f'{query} unveiled its latest innovation in a press conference yesterday.',
                'source': {'name': 'TechCrunch'},
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat() + 'Z',
                'url': 'https://example.com/article2'
            },
            {
                'title': f'Analysts Upgrade {query} Stock Rating',
                'content': f'Major investment firms have upgraded their outlook for {query} citing strong fundamentals.',
                'source': {'name': 'Bloomberg'},
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat() + 'Z',
                'url': 'https://example.com/article3'
            }
        ]
        return sample_articles

class DataManager:
    """Manages data storage and retrieval"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_stock_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save stock data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Stock data saved to {filepath}")
        return filepath
    
    def save_news_data(self, articles: List[NewsArticle], filename: str = None) -> str:
        """Save news data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_data_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert NewsArticle objects to dictionaries
        articles_dict = []
        for article in articles:
            articles_dict.append({
                'title': article.title,
                'content': article.content,
                'source': article.source,
                'published_at': article.published_at.isoformat(),
                'url': article.url,
                'symbol': article.symbol
            })
        
        with open(filepath, 'w') as f:
            json.dump(articles_dict, f, indent=2)
        
        logger.info(f"News data saved to {filepath}")
        return filepath
    
    def load_stock_data(self, filename: str) -> pd.DataFrame:
        """Load stock data from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        logger.info(f"Stock data loaded from {filepath}")
        return data
    
    def load_news_data(self, filename: str) -> List[NewsArticle]:
        """Load news data from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r') as f:
            articles_dict = json.load(f)
        
        articles = []
        for article_data in articles_dict:
            article = NewsArticle(
                title=article_data['title'],
                content=article_data['content'],
                source=article_data['source'],
                published_at=datetime.fromisoformat(article_data['published_at']),
                url=article_data['url'],
                symbol=article_data['symbol']
            )
            articles.append(article)
        
        logger.info(f"News data loaded from {filepath}")
        return articles

def main():
    """Main function to demonstrate data collection"""
    print("🚀 Market Pulse - Data Collection Demo")
    print("="*50)
    
    # Initialize collectors
    stock_collector = StockDataCollector()
    news_collector = NewsDataCollector()
    data_manager = DataManager()
    
    # Collect stock data
    print("\n📈 Collecting Stock Data...")
    stock_data = stock_collector.get_multiple_stocks_data(period="6mo")
    
    if not stock_data.empty:
        # Add technical indicators
        print("🔧 Calculating Technical Indicators...")
        stock_data_with_indicators = stock_collector.calculate_technical_indicators(stock_data)
        
        # Save stock data
        stock_file = data_manager.save_stock_data(stock_data_with_indicators)
        print(f"✅ Stock data saved: {stock_file}")
        print(f"📊 Data shape: {stock_data_with_indicators.shape}")
        print(f"📅 Date range: {stock_data_with_indicators['Date'].min()} to {stock_data_with_indicators['Date'].max()}")
    
    # Collect news data
    print("\n📰 Collecting News Data...")
    all_news = []
    
    for symbol in stock_collector.symbols:
        print(f"  Fetching news for {symbol}...")
        news_articles = news_collector.get_stock_news(symbol, days_back=30)
        all_news.extend(news_articles)
        time.sleep(1)  # Rate limiting
    
    if all_news:
        # Save news data
        news_file = data_manager.save_news_data(all_news)
        print(f"✅ News data saved: {news_file}")
        print(f"📰 Total articles: {len(all_news)}")
        
        # Show sample articles
        print("\n📖 Sample News Headlines:")
        for i, article in enumerate(all_news[:5]):
            print(f"  {i+1}. [{article.symbol}] {article.title[:80]}...")
    
    print("\n🎉 Data collection completed successfully!")
    print("Next steps: Run sentiment analysis and model training")

if __name__ == "__main__":
    main()
