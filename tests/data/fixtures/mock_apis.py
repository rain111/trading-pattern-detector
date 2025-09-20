"""
Mock API implementations for testing network scenarios and API failures.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import random
from unittest.mock import AsyncMock, MagicMock


class MockYahooFinanceAPI:
    """Mock Yahoo Finance API for testing various scenarios."""

    def __init__(self, failure_rate: float = 0.0, latency_range: tuple = (0.1, 0.5)):
        """
        Initialize mock API with configurable failure rate and latency.

        Args:
            failure_rate: Probability of API failure (0.0 to 1.0)
            latency_range: Range of latency in seconds
        """
        self.failure_rate = failure_rate
        self.latency_range = latency_range
        self.call_count = 0
        self.rate_limit_delay = 0.1  # 100ms between calls for rate limiting

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Mock implementation of Yahoo Finance data fetching.

        Args:
            symbol: Stock symbol to fetch data for
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with mock market data
        """
        self.call_count += 1

        # Simulate rate limiting
        if self.call_count > 1:
            await asyncio.sleep(self.rate_limit_delay)

        # Simulate network latency
        latency = random.uniform(*self.latency_range)
        await asyncio.sleep(latency)

        # Simulate API failures
        if random.random() < self.failure_rate:
            raise Exception(f"API failed for {symbol} (failure rate: {self.failure_rate})")

        # Generate mock data
        data = await self._generate_mock_data(symbol, start_date, end_date)

        # Simulate occasional timeout
        if random.random() < 0.05:  # 5% chance of timeout
            await asyncio.sleep(10)  # Simulate timeout

        return data

    async def _generate_mock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate realistic mock market data."""
        # Create date range
        date_range = pd.date_range(start_date, end_date, freq='D')
        date_range = date_range[date_range.weekday < 5]  # Remove weekends

        if len(date_range) == 0:
            return pd.DataFrame()

        # Generate price based on symbol characteristics
        base_price = 100.0 + (ord(symbol[0]) - ord('A')) * 10
        prices = [base_price]

        # Generate price movements
        for i in range(1, len(date_range)):
            # Add some trend based on symbol
            trend_factor = 0.001 * (ord(symbol[-1]) - ord('A')) / 10
            volatility = 0.015 + 0.005 * (len(symbol) / 10)

            change = np.random.normal(trend_factor, volatility)
            prices.append(prices[-1] * (1 + change))

        # Create OHLC data
        data = pd.DataFrame({
            'open': prices[:-1],
            'close': prices[1:],
            'high': prices[:-1] * (1 + np.random.uniform(0, 0.01, len(prices)-1)),
            'low': prices[:-1] * (1 - np.random.uniform(0, 0.01, len(prices)-1)),
            'volume': np.random.uniform(1000000, 5000000, len(prices)-1)
        }, index=date_range[:-1])

        # Ensure OHLC relationships
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)

        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get API call statistics."""
        return {
            'call_count': self.call_count,
            'failure_rate': self.failure_rate,
            'last_call_latency': f"{random.uniform(*self.latency_range):.3f}s"
        }


class MockAlphaVantageAPI:
    """Mock Alpha Vantage API for testing alternative data sources."""

    def __init__(self, api_key: str = "test_key", rate_limit: int = 5):
        """
        Initialize mock Alpha Vantage API.

        Args:
            api_key: Mock API key
            rate_limit: Calls per minute
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.call_count = 0
        self.minute_start = datetime.now()
        self.minute_calls = 0

    async def fetch_data(self, symbol: str, function: str = "TIME_SERIES_DAILY") -> pd.DataFrame:
        """
        Mock implementation of Alpha Vantage data fetching.

        Args:
            symbol: Stock symbol
            function: API function to call

        Returns:
            DataFrame with mock data
        """
        # Check rate limiting
        now = datetime.now()
        if (now - self.minute_start).total_seconds() >= 60:
            self.minute_start = now
            self.minute_calls = 0

        if self.minute_calls >= self.rate_limit:
            # Simulate rate limit error
            await asyncio.sleep(60 - (now - self.minute_start).total_seconds())
            self.minute_start = datetime.now()
            self.minute_calls = 0

        self.minute_calls += 1
        self.call_count += 1

        # Simulate API response delay
        await asyncio.sleep(0.2)

        # Generate different data based on function
        if function == "TIME_SERIES_INTRADAY":
            return await self._generate_intraday_data(symbol)
        else:
            return await self._generate_daily_data(symbol)

    async def _generate_daily_data(self, symbol: str) -> pd.DataFrame:
        """Generate daily time series data."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        dates = dates[dates.weekday < 5]

        base_price = 150.0 + (ord(symbol[0]) - ord('A')) * 5
        prices = [base_price]

        for i in range(1, len(dates)):
            change = np.random.normal(0.0008, 0.012)
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame({
            '1. open': prices[:-1],
            '2. high': prices[:-1] * (1 + np.random.uniform(0, 0.008, len(prices)-1)),
            '3. low': prices[:-1] * (1 - np.random.uniform(0, 0.008, len(prices)-1)),
            '4. close': prices[1:],
            '5. volume': np.random.uniform(2000000, 8000000, len(prices)-1)
        }, index=dates[:-1])

        # Rename columns to standard format
        data = data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        return data

    async def _generate_intraday_data(self, symbol: str) -> pd.DataFrame:
        """Generate intraday time series data."""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='H')

        base_price = 200.0
        prices = [base_price]

        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.003)
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame({
            '1. open': prices[:-1],
            '2. high': prices[:-1] * (1 + np.random.uniform(0, 0.002, len(prices)-1)),
            '3. low': prices[:-1] * (1 - np.random.uniform(0, 0.002, len(prices)-1)),
            '4. close': prices[1:],
            '5. volume': np.random.uniform(500000, 2000000, len(prices)-1)
        }, index=dates[:-1])

        # Rename columns
        data = data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        return data


class MockAPIMix:
    """Mock API that simulates mixed scenarios (success, failure, timeout)."""

    def __init__(self, scenarios: List[str] = None):
        """
        Initialize with specific scenarios.

        Args:
            scenarios: List of scenario names to simulate
        """
        self.scenarios = scenarios or ['success', 'failure', 'timeout', 'rate_limit']
        self.scenario_index = 0
        self.call_count = 0

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch data with mixed scenarios.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame or raises exception based on scenario
        """
        self.call_count += 1

        # Get current scenario
        scenario = self.scenarios[self.scenario_index % len(self.scenarios)]
        self.scenario_index += 1

        # Execute scenario
        if scenario == 'failure':
            raise Exception(f"API connection failed for {symbol}")
        elif scenario == 'timeout':
            await asyncio.sleep(5)  # Simulate timeout
            return await self._generate_mock_data(symbol, start_date, end_date)
        elif scenario == 'rate_limit':
            await asyncio.sleep(2)  # Simulate rate limit delay
            return await self._generate_mock_data(symbol, start_date, end_date)
        else:  # success
            await asyncio.sleep(0.1)  # Normal latency
            return await self._generate_mock_data(symbol, start_date, end_date)

    async def _generate_mock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate mock data for mixed scenarios."""
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]

        if len(dates) == 0:
            return pd.DataFrame()

        base_price = 120.0
        prices = [base_price]

        for i in range(1, len(dates)):
            change = np.random.normal(0.001, 0.01)
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame({
            'open': prices[:-1],
            'close': prices[1:],
            'high': prices[:-1] * (1 + np.random.uniform(0, 0.008, len(prices)-1)),
            'low': prices[:-1] * (1 - np.random.uniform(0, 0.008, len(prices)-1)),
            'volume': np.random.uniform(1500000, 6000000, len(prices)-1)
        }, index=dates[:-1])

        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)

        return data


class MockErrorResponseAPI:
    """Mock API that returns various error responses."""

    ERROR_TYPES = [
        "NetworkError",
        "TimeoutError",
        "RateLimitError",
        "AuthenticationError",
        "InvalidSymbolError",
        "ServerError",
        "DataCorruptionError"
    ]

    def __init__(self):
        self.error_calls = 0

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data that always returns an error."""
        self.error_calls += 1

        # Cycle through different error types
        error_type = self.ERROR_TYPES[self.error_calls % len(self.ERROR_TYPES)]

        # Simulate different error behaviors
        if error_type == "NetworkError":
            await asyncio.sleep(0.1)
            raise aiohttp.ClientError("Network connection failed")
        elif error_type == "TimeoutError":
            await asyncio.sleep(3)
            raise asyncio.TimeoutError("Request timeout")
        elif error_type == "RateLimitError":
            await asyncio.sleep(1)
            raise Exception(f"Rate limit exceeded for {symbol}")
        elif error_type == "AuthenticationError":
            raise Exception(f"Invalid API key for {symbol}")
        elif error_type == "InvalidSymbolError":
            raise Exception(f"Invalid symbol: {symbol}")
        elif error_type == "ServerError":
            raise Exception(f"Server error (500) for {symbol}")
        elif error_type == "DataCorruptionError":
            # Return malformed data instead of raising exception
            malformed_data = pd.DataFrame({
                'open': ['invalid', 'data'],
                'high': [101, 102],
                'low': [99, 100],
                'close': [100.5, 101.5],
                'volume': [1000000, 1100000]
            }, index=pd.date_range(start_date, periods=2, freq='D'))
            return malformed_data

        # Default error
        raise Exception(f"Unknown error: {error_type}")


class MockAPIWithRetries:
    """Mock API that simulates retry behavior."""

    def __init__(self, success_on_nth_attempt: int = 3):
        """
        Initialize API with configurable retry behavior.

        Args:
            success_on_nth_attempt: Which attempt will succeed
        """
        self.success_on_nth_attempt = success_on_nth_attempt
        self.attempt_count = 0

    async def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data that fails until the nth attempt."""
        self.attempt_count += 1

        if self.attempt_count < self.success_on_nth_attempt:
            # Simulate retryable error
            raise Exception(f"Temporary error for {symbol} (attempt {self.attempt_count})")

        # Success after nth attempt
        return await self._generate_mock_data(symbol, start_date, end_date)

    async def _generate_mock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate mock data for successful retry."""
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]

        if len(dates) == 0:
            return pd.DataFrame()

        base_price = 180.0
        prices = [base_price]

        for i in range(1, len(dates)):
            change = np.random.normal(0.0005, 0.008)
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame({
            'open': prices[:-1],
            'close': prices[1:],
            'high': prices[:-1] * (1 + np.random.uniform(0, 0.006, len(prices)-1)),
            'low': prices[:-1] * (1 - np.random.uniform(0, 0.006, len(prices)-1)),
            'volume': np.random.uniform(1200000, 4500000, len(prices)-1)
        }, index=dates[:-1])

        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)

        return data


# Factory function to create mock APIs based on test requirements
def create_mock_api(api_type: str, **kwargs) -> Any:
    """
    Factory function to create mock APIs for different test scenarios.

    Args:
        api_type: Type of mock API to create
        **kwargs: Additional arguments for API initialization

    Returns:
        Mock API instance
    """
    api_factories = {
        'yahoo_finance': MockYahooFinanceAPI,
        'alpha_vantage': MockAlphaVantageAPI,
        'mixed_scenarios': MockAPIMix,
        'error_responses': MockErrorResponseAPI,
        'with_retries': MockAPIWithRetries
    }

    if api_type not in api_factories:
        raise ValueError(f"Unknown API type: {api_type}")

    return api_factories[api_type](**kwargs)


if __name__ == "__main__":
    # Test mock API implementations
    async def test_mock_apis():
        print("Testing mock API implementations...")

        # Test Yahoo Finance mock
        yahoo_api = MockYahooFinanceAPI(failure_rate=0.1, latency_range=(0.05, 0.2))
        try:
            data = await yahoo_api.fetch_data("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 5))
            print(f"✅ Yahoo Finance mock: {len(data)} rows")
        except Exception as e:
            print(f"❌ Yahoo Finance mock error: {e}")

        # Test Alpha Vantage mock
        alpha_api = MockAlphaVantageAPI()
        try:
            data = await alpha_api.fetch_data("MSFT", datetime(2023, 1, 1), datetime(2023, 1, 5))
            print(f"✅ Alpha Vantage mock: {len(data)} rows")
        except Exception as e:
            print(f"❌ Alpha Vantage mock error: {e}")

        # Test mixed scenarios
        mixed_api = MockAPIMix(['success', 'failure', 'timeout'])
        try:
            data = await mixed_api.fetch_data("GOOGL", datetime(2023, 1, 1), datetime(2023, 1, 3))
            print(f"✅ Mixed scenarios mock: {len(data)} rows")
        except Exception as e:
            print(f"❌ Mixed scenarios mock error: {e}")

        # Test error responses
        error_api = MockErrorResponseAPI()
        try:
            data = await error_api.fetch_data("INVALID", datetime(2023, 1, 1), datetime(2023, 1, 3))
            print(f"❌ Error API should have failed but returned: {len(data)} rows")
        except Exception as e:
            print(f"✅ Error API correctly failed: {type(e).__name__}")

        print("Mock API testing completed!")

    # Run the test
    import asyncio
    asyncio.run(test_mock_apis())