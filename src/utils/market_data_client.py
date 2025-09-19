import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import yfinance as yf
from .data_preprocessor import DataPreprocessor


class MarketDataClient:
    """Market data fetching and processing"""
    
    def __init__(self, data_source: str = 'yahoo'):
        self.data_source = data_source
        self.preprocessor = DataPreprocessor()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_data(self, symbol: str, period: str = '1y', 
                  timeframe: str = '1d') -> pd.DataFrame:
        """Fetch market data for a symbol"""
        try:
            if self.data_source == 'yahoo':
                return self._fetch_from_yahoo(symbol, period, timeframe)
            elif self.data_source == 'alpha_vantage':
                return self._fetch_from_alpha_vantage(symbol, period, timeframe)
            else:
                self.logger.error(f"Unsupported data source: {self.data_source}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_yahoo(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=timeframe)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make datetime a column
            data.reset_index(inplace=True)
            data.set_index('Date', inplace=True)
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching from Yahoo for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_alpha_vantage(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage (placeholder implementation)"""
        # This would require an Alpha Vantage API key
        self.logger.warning("Alpha Vantage integration not implemented")
        return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], 
                             period: str = '1y',
                             timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, period, timeframe)
                if not data.empty:
                    data = self.preprocessor.clean_data(data)
                    data_dict[symbol] = data
                else:
                    self.logger.warning(f"No data fetched for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data_dict
    
    def get_universe_data(self, universe_file: str) -> Dict[str, pd.DataFrame]:
        """Get data for universe of symbols from file"""
        try:
            # Read universe from file
            with open(universe_file, 'r') as f:
                symbols = [line.strip() for line in f.readlines() if line.strip()]
            
            return self.fetch_multiple_symbols(symbols)
        except Exception as e:
            self.logger.error(f"Error getting universe data: {e}")
            return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists and can be fetched"""
        try:
            data = self.fetch_data(symbol, period='1d', timeframe='1m')
            return not data.empty
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic information about a symbol"""
        try:
            if self.data_source == 'yahoo':
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    'symbol': symbol,
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'currency': info.get('currency', ''),
                    'exchange': info.get('exchange', '')
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def get_historical_data_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for a specific date range"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            data.reset_index(inplace=True)
            data.set_index('Date', inplace=True)
            
            return self.preprocessor.clean_data(data)
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str) -> Dict[str, float]:
        """Get real-time data for a symbol"""
        try:
            if self.data_source == 'yahoo':
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                return {
                    'symbol': symbol,
                    'price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'timestamp': datetime.now()
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {e}")
            return {}
    
    def create_market_universe(self, symbols: List[str], 
                              output_file: str = 'market_universe.csv') -> None:
        """Create a market universe file"""
        try:
            with open(output_file, 'w') as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            self.logger.info(f"Market universe created: {output_file}")
        except Exception as e:
            self.logger.error(f"Error creating market universe: {e}")
    
    def get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes for the data source"""
        if self.data_source == 'yahoo':
            return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        else:
            return ['1d']
    
    def get_supported_periods(self) -> List[str]:
        """Get supported periods for the data source"""
        if self.data_source == 'yahoo':
            return ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        else:
            return ['1y']
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics"""
        try:
            quality_metrics = {
                'total_rows': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'date_range': {
                    'start': data.index.min(),
                    'end': data.index.max(),
                    'total_days': (data.index.max() - data.index.min()).days
                },
                'price_quality': {
                    'min_price': data['close'].min(),
                    'max_price': data['close'].max(),
                    'price_range': data['close'].max() - data['close'].min()
                },
                'volume_quality': {
                    'avg_volume': data['volume'].mean(),
                    'total_volume': data['volume'].sum(),
                    'volume_zero_count': (data['volume'] == 0).sum()
                }
            }
            
            # Check for potential data issues
            issues = []
            if quality_metrics['duplicate_rows'] > 0:
                issues.append(f"Found {quality_metrics['duplicate_rows']} duplicate rows")
            
            if quality_metrics['volume_quality']['volume_zero_count'] > 0:
                issues.append(f"Found {quality_metrics['volume_quality']['volume_zero_count']} zero volume days")
            
            # Check for unusual price gaps
            price_gaps = data['close'].pct_change().abs()
            large_gaps = price_gaps[price_gaps > 0.1]  # 10% gap threshold
            if len(large_gaps) > 0:
                issues.append(f"Found {len(large_gaps)} price gaps > 10%")
            
            quality_metrics['issues'] = issues
            
            return quality_metrics
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {}