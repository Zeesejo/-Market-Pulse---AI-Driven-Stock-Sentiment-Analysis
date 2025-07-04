"""
Real-time News Data Collection with APIs
Extends the sample news generation with actual news feeds
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Optional

class RealNewsCollector:
    """
    Collects real financial news from multiple APIs
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize with API keys for various news services
        
        Args:
            api_keys: Dictionary containing API keys for:
                - 'newsapi': NewsAPI.org key
                - 'finnhub': Finnhub.io key
                - 'alpha_vantage': Alpha Vantage key
        """
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        
    def get_newsapi_articles(self, 
                           symbols: List[str], 
                           days_back: int = 30,
                           language: str = 'en') -> List[Dict]:
        """
        Fetch news from NewsAPI.org
        
        Free tier: 100 requests/day, 1000 articles/request
        """
        if 'newsapi' not in self.api_keys:
            print("⚠️ NewsAPI key not provided. Using sample data.")
            return []
            
        articles = []
        base_url = "https://newsapi.org/v2/everything"
        
        for symbol in symbols:
            # Get company name for better search
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft', 
                'GOOGL': 'Google',
                'TSLA': 'Tesla',
                'AMZN': 'Amazon'
            }
            
            query = f"{company_names.get(symbol, symbol)} stock OR {symbol}"
            
            params = {
                'q': query,
                'language': language,
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'apiKey': self.api_keys['newsapi']
            }
            
            try:
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for article in data.get('articles', []):
                    articles.append({
                        'symbol': symbol,
                        'date': pd.to_datetime(article['publishedAt']).date(),
                        'headline': article['title'],
                        'content': article['description'] or article['title'],
                        'source': article['source']['name'],
                        'url': article['url']
                    })
                    
                print(f"✅ Fetched {len(data.get('articles', []))} articles for {symbol}")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error fetching news for {symbol}: {e}")
                
        return articles
    
    def get_finnhub_news(self, 
                        symbols: List[str], 
                        days_back: int = 30) -> List[Dict]:
        """
        Fetch news from Finnhub.io
        
        Free tier: 60 calls/minute
        """
        if 'finnhub' not in self.api_keys:
            print("⚠️ Finnhub API key not provided. Using sample data.")
            return []
            
        articles = []
        base_url = "https://finnhub.io/api/v1/company-news"
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_keys['finnhub']
            }
            
            try:
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for article in data:
                    articles.append({
                        'symbol': symbol,
                        'date': pd.to_datetime(article['datetime'], unit='s').date(),
                        'headline': article['headline'],
                        'content': article['summary'],
                        'source': article['source'],
                        'url': article['url']
                    })
                    
                print(f"✅ Fetched {len(data)} articles for {symbol} from Finnhub")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error fetching Finnhub news for {symbol}: {e}")
                
        return articles
    
    def collect_real_news(self, 
                         symbols: List[str], 
                         days_back: int = 30) -> pd.DataFrame:
        """
        Collect news from all available sources
        """
        print(f"📰 Collecting real news for {len(symbols)} symbols...")
        
        all_articles = []
        
        # Try each API source
        if 'newsapi' in self.api_keys:
            all_articles.extend(self.get_newsapi_articles(symbols, days_back))
            
        if 'finnhub' in self.api_keys:
            all_articles.extend(self.get_finnhub_news(symbols, days_back))
        
        if not all_articles:
            print("⚠️ No API keys provided. Falling back to sample data generation.")
            return self._generate_fallback_news(symbols, days_back)
            
        # Convert to DataFrame and clean
        news_df = pd.DataFrame(all_articles)
        news_df = news_df.drop_duplicates(subset=['symbol', 'headline', 'date'])
        news_df = news_df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        print(f"✅ Collected {len(news_df)} unique articles")
        return news_df
    
    def _generate_fallback_news(self, symbols: List[str], days_back: int) -> pd.DataFrame:
        """
        Generate sample news when no API keys are available
        """
        from src.data_collection import generate_sample_news
        print("🔄 Generating sample news data...")
        return generate_sample_news(symbols, days_back)

# Example usage with API keys
def setup_real_news_collection():
    """
    Setup real news collection with API keys
    """
    
    # Example API key setup (replace with your actual keys)
    api_keys = {
        # Get free key from: https://newsapi.org/
        'newsapi': 'YOUR_NEWSAPI_KEY_HERE',
        
        # Get free key from: https://finnhub.io/
        'finnhub': 'YOUR_FINNHUB_KEY_HERE',
        
        # Optional: Alpha Vantage for additional news
        # 'alpha_vantage': 'YOUR_ALPHA_VANTAGE_KEY_HERE'
    }
    
    # Remove keys that are not set
    api_keys = {k: v for k, v in api_keys.items() if v != f'YOUR_{k.upper()}_KEY_HERE'}
    
    collector = RealNewsCollector(api_keys)
    return collector

# Integration example
if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    collector = setup_real_news_collection()
    news_df = collector.collect_real_news(symbols, days_back=30)
    
    print("\n📊 News Collection Summary:")
    print(f"Total articles: {len(news_df)}")
    print(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    print("Articles per symbol:")
    print(news_df['symbol'].value_counts())
