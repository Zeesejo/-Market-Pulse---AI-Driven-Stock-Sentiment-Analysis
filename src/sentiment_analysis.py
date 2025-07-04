"""
Market Pulse - Sentiment Analysis Module

This module performs sentiment analysis on financial news articles using
multiple approaches including VADER, TextBlob, and transformer models.
"""

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text cleaning and preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Financial keywords that might be important for sentiment
        self.financial_keywords = {
            'positive': ['profit', 'revenue', 'growth', 'gain', 'rise', 'increase', 
                        'surge', 'boost', 'strong', 'beat', 'exceed', 'outperform',
                        'upgrade', 'bullish', 'buy', 'target', 'optimistic'],
            'negative': ['loss', 'decline', 'fall', 'drop', 'decrease', 'plunge',
                        'weak', 'miss', 'underperform', 'downgrade', 'bearish',
                        'sell', 'pessimistic', 'concern', 'worry', 'risk']
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media data)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep important financial symbols
        text = re.sub(r'[^\w\s$%]', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str) -> Dict[str, int]:
        """
        Extract financial keywords from text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with keyword counts
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.financial_keywords['positive'] 
                           if keyword in text_lower)
        negative_count = sum(1 for keyword in self.financial_keywords['negative'] 
                           if keyword in text_lower)
        
        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'keyword_ratio': positive_count - negative_count
        }
    
    def get_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract additional text features for sentiment analysis
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with text features
        """
        if not text:
            return {
                'text_length': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'uppercase_ratio': 0
            }
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            'text_length': len(text),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

class SentimentAnalyzer:
    """Main sentiment analysis class using multiple methods"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer model (optional, requires transformers library)
        self.transformer_model = None
        try:
            from transformers import pipeline
            self.transformer_model = pipeline("sentiment-analysis", 
                                             model="ProsusAI/finbert",
                                             return_all_scores=True)
            logger.info("FinBERT transformer model loaded successfully")
        except ImportError:
            logger.warning("Transformers library not available. Using VADER and TextBlob only.")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {str(e)}")
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with VADER sentiment scores
        """
        if not text:
            return {'vader_positive': 0, 'vader_negative': 0, 'vader_neutral': 0, 'vader_compound': 0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with TextBlob sentiment scores
        """
        if not text:
            return {'textblob_polarity': 0, 'textblob_subjectivity': 0}
        
        blob = TextBlob(text)
        
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis using transformer model (FinBERT)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with transformer sentiment scores
        """
        if not self.transformer_model or not text:
            return {'finbert_positive': 0, 'finbert_negative': 0, 'finbert_neutral': 0}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            results = self.transformer_model(text)
            
            # Convert results to dictionary
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[f'finbert_{label}'] = score
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"Error in transformer analysis: {str(e)}")
            return {'finbert_positive': 0, 'finbert_negative': 0, 'finbert_neutral': 0}
    
    def get_composite_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get composite sentiment score from all methods
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with all sentiment scores and composite score
        """
        if not text:
            return self._empty_sentiment_result()
        
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Get sentiment scores from all methods
        vader_scores = self.analyze_with_vader(text)  # Use original text for VADER
        textblob_scores = self.analyze_with_textblob(cleaned_text)
        transformer_scores = self.analyze_with_transformer(cleaned_text)
        
        # Extract keywords and text features
        keyword_features = self.preprocessor.extract_keywords(text)
        text_features = self.preprocessor.get_text_features(text)
        
        # Calculate composite sentiment
        composite_positive = np.mean([
            vader_scores['vader_positive'],
            max(0, textblob_scores['textblob_polarity']),  # Convert to 0-1 range
            transformer_scores.get('finbert_positive', 0)
        ])
        
        composite_negative = np.mean([
            vader_scores['vader_negative'],
            max(0, -textblob_scores['textblob_polarity']),  # Convert to 0-1 range
            transformer_scores.get('finbert_negative', 0)
        ])
        
        composite_neutral = np.mean([
            vader_scores['vader_neutral'],
            1 - abs(textblob_scores['textblob_polarity']),  # Neutral when polarity near 0
            transformer_scores.get('finbert_neutral', 0)
        ])
        
        # Overall sentiment score (-1 to 1)
        overall_sentiment = composite_positive - composite_negative
        
        # Adjust based on keyword analysis
        keyword_adjustment = keyword_features['keyword_ratio'] * 0.1
        overall_sentiment += keyword_adjustment
        
        # Ensure scores are in valid range
        overall_sentiment = max(-1, min(1, overall_sentiment))
        
        # Combine all results
        result = {
            **vader_scores,
            **textblob_scores,
            **transformer_scores,
            **keyword_features,
            **text_features,
            'composite_positive': composite_positive,
            'composite_negative': composite_negative,
            'composite_neutral': composite_neutral,
            'overall_sentiment': overall_sentiment,
            'sentiment_magnitude': abs(overall_sentiment)
        }
        
        return result
    
    def _empty_sentiment_result(self) -> Dict[str, float]:
        """Return empty sentiment result for invalid input"""
        return {
            'vader_positive': 0, 'vader_negative': 0, 'vader_neutral': 0, 'vader_compound': 0,
            'textblob_polarity': 0, 'textblob_subjectivity': 0,
            'finbert_positive': 0, 'finbert_negative': 0, 'finbert_neutral': 0,
            'positive_keywords': 0, 'negative_keywords': 0, 'keyword_ratio': 0,
            'text_length': 0, 'sentence_count': 0, 'avg_sentence_length': 0,
            'exclamation_count': 0, 'question_count': 0, 'uppercase_ratio': 0,
            'composite_positive': 0, 'composite_negative': 0, 'composite_neutral': 0,
            'overall_sentiment': 0, 'sentiment_magnitude': 0
        }
    
    def categorize_sentiment(self, sentiment_score: float) -> str:
        """
        Categorize sentiment score into labels
        
        Args:
            sentiment_score: Overall sentiment score (-1 to 1)
        
        Returns:
            Sentiment category
        """
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'

class NewsAnalyzer:
    """Analyzes news data and creates sentiment features"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_news_batch(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of news articles
        
        Args:
            news_data: List of news article dictionaries
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        logger.info(f"Analyzing sentiment for {len(news_data)} articles...")
        
        for i, article in enumerate(news_data):
            try:
                # Combine title and content for analysis
                full_text = f"{article.get('title', '')} {article.get('content', '')}"
                
                # Get sentiment analysis
                sentiment_result = self.sentiment_analyzer.get_composite_sentiment(full_text)
                
                # Add article metadata
                result = {
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'published_at': article.get('published_at', ''),
                    'symbol': article.get('symbol', ''),
                    'url': article.get('url', ''),
                    **sentiment_result
                }
                
                # Add sentiment category
                result['sentiment_category'] = self.sentiment_analyzer.categorize_sentiment(
                    sentiment_result['overall_sentiment']
                )
                
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(news_data)} articles")
                    
            except Exception as e:
                logger.warning(f"Error processing article {i}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Convert published_at to datetime
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'])
                df['date'] = df['published_at'].dt.date
        
        logger.info(f"Sentiment analysis completed for {len(df)} articles")
        return df
    
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment scores by symbol and date
        
        Args:
            sentiment_df: DataFrame with individual article sentiments
        
        Returns:
            DataFrame with daily aggregated sentiment scores
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Group by symbol and date
        daily_sentiment = sentiment_df.groupby(['symbol', 'date']).agg({
            'overall_sentiment': ['mean', 'std', 'count'],
            'sentiment_magnitude': ['mean', 'max'],
            'composite_positive': 'mean',
            'composite_negative': 'mean',
            'composite_neutral': 'mean',
            'vader_compound': 'mean',
            'textblob_polarity': 'mean',
            'positive_keywords': 'sum',
            'negative_keywords': 'sum',
            'text_length': 'mean'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                 for col in daily_sentiment.columns]
        
        daily_sentiment = daily_sentiment.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'overall_sentiment_mean': 'daily_sentiment',
            'overall_sentiment_std': 'sentiment_volatility',
            'overall_sentiment_count': 'news_volume',
            'sentiment_magnitude_mean': 'avg_sentiment_strength',
            'sentiment_magnitude_max': 'max_sentiment_strength'
        }
        
        daily_sentiment.rename(columns=column_mapping, inplace=True)
        
        # Add sentiment trend indicators
        daily_sentiment['sentiment_trend'] = daily_sentiment.groupby('symbol')['daily_sentiment'].diff()
        
        # Add sentiment category for daily aggregation
        daily_sentiment['daily_sentiment_category'] = daily_sentiment['daily_sentiment'].apply(
            self.sentiment_analyzer.categorize_sentiment
        )
        
        logger.info(f"Daily sentiment aggregation completed for {len(daily_sentiment)} symbol-date pairs")
        return daily_sentiment

def main():
    """Main function to demonstrate sentiment analysis"""
    print("🧠 Market Pulse - Sentiment Analysis Demo")
    print("="*50)
    
    # Sample news data for testing
    sample_news = [
        {
            'title': 'Apple Reports Record-Breaking Q4 Revenue',
            'content': 'Apple Inc. announced strong quarterly results with revenue exceeding expectations by 12%. The company showed remarkable growth in all product categories.',
            'source': 'Financial Times',
            'published_at': '2024-07-03',
            'symbol': 'AAPL'
        },
        {
            'title': 'Tesla Stock Plunges on Production Concerns',
            'content': 'Tesla shares fell 8% in after-hours trading following reports of production delays and supply chain issues affecting the Model 3.',
            'source': 'Reuters',
            'published_at': '2024-07-03',
            'symbol': 'TSLA'
        },
        {
            'title': 'Microsoft Azure Shows Steady Growth',
            'content': 'Microsoft reported continued expansion in its cloud services division with Azure revenue growing 25% year-over-year.',
            'source': 'TechCrunch',
            'published_at': '2024-07-02',
            'symbol': 'MSFT'
        }
    ]
    
    # Initialize analyzer
    news_analyzer = NewsAnalyzer()
    
    # Analyze sentiment
    print("\n🔍 Analyzing News Sentiment...")
    sentiment_df = news_analyzer.analyze_news_batch(sample_news)
    
    if not sentiment_df.empty:
        print(f"✅ Sentiment analysis completed for {len(sentiment_df)} articles")
        
        # Display results
        print("\n📊 Sentiment Analysis Results:")
        for _, row in sentiment_df.iterrows():
            print(f"  [{row['symbol']}] {row['title'][:60]}...")
            print(f"    Sentiment: {row['overall_sentiment']:.3f} ({row['sentiment_category']})")
            print(f"    VADER: {row['vader_compound']:.3f}, TextBlob: {row['textblob_polarity']:.3f}")
            print()
        
        # Aggregate daily sentiment
        print("📈 Aggregating Daily Sentiment...")
        daily_sentiment = news_analyzer.aggregate_daily_sentiment(sentiment_df)
        
        if not daily_sentiment.empty:
            print(f"✅ Daily sentiment aggregation completed")
            print("\n📅 Daily Sentiment Summary:")
            for _, row in daily_sentiment.iterrows():
                print(f"  {row['symbol']} ({row['date']}): "
                      f"Sentiment {row['daily_sentiment']:.3f} "
                      f"({row['daily_sentiment_category']}) "
                      f"[{row['news_volume']} articles]")
    
    print("\n🎉 Sentiment analysis demo completed!")
    print("Next steps: Integrate with stock price data for prediction modeling")

if __name__ == "__main__":
    main()
