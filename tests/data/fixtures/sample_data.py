"""
Test data fixtures for enhanced data management system testing.
Provides realistic market data samples for comprehensive testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def create_realistic_market_data(
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    symbol: str = 'AAPL'
) -> pd.DataFrame:
    """
    Create realistic market data with proper OHLCV relationships.

    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        symbol: Stock symbol for data labeling

    Returns:
        DataFrame with realistic market data
    """
    # Create date range
    dates = pd.date_range(start_date, end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible tests

    # Starting price
    current_price = 100.0

    # Price changes with mean-reverting characteristics
    price_changes = np.random.normal(0.0005, 0.015, len(dates))

    # Apply some autocorrelation for realistic patterns
    for i in range(1, len(price_changes)):
        price_changes[i] += 0.3 * price_changes[i-1]

    # Generate price series
    prices = [current_price]
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    prices = np.array(prices)

    # Create OHLC data with realistic relationships
    data = pd.DataFrame({
        'open': prices[:-1],
        'close': prices[1:],
        'high': prices[:-1] * (1 + np.random.uniform(0, 0.008, len(prices)-1)),
        'low': prices[:-1] * (1 - np.random.uniform(0, 0.008, len(prices)-1)),
        'volume': np.random.uniform(5000000, 20000000, len(prices)-1)
    }, index=dates[:-1])

    # Ensure OHLC relationships (high >= max(open, close), low <= min(open, close))
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)

    # Add symbol and other metadata
    data['symbol'] = symbol
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # Handle missing values
    data = data.dropna()

    return data


def create_pattern_data(
    pattern_type: str = 'flag',
    start_date: str = '2023-01-01',
    duration_days: int = 30
) -> pd.DataFrame:
    """
    Create data with specific chart patterns for pattern detection testing.

    Args:
        pattern_type: Type of pattern to generate ('flag', 'cup_handle', 'triangle', etc.)
        start_date: Start date for pattern
        duration_days: Duration of pattern in days

    Returns:
        DataFrame with pattern-specific data
    """
    dates = pd.date_range(start_date, periods=duration_days, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    base_price = 100.0

    if pattern_type == 'flag':
        # Create flag pattern (consolidation after upward trend)
        pattern_data = []

        # Upward trend (5 days)
        trend_days = min(5, len(dates) // 4)
        for i, date in enumerate(dates[:trend_days]):
            price = base_price * (1 + 0.02 * i)
            pattern_data.append({
                'date': date,
                'open': price * 0.995,
                'high': price * 1.005,
                'low': price * 0.995,
                'close': price,
                'volume': 15000000 + i * 1000000
            })

        # Flag consolidation (10 days) - sideways movement
        consolidation_days = min(10, len(dates) // 3)
        base_price = base_price * (1 + 0.02 * trend_days)
        for i, date in enumerate(dates[trend_days: trend_days + consolidation_days]):
            price_variation = np.random.uniform(-0.005, 0.005)
            price = base_price * (1 + price_variation)
            pattern_data.append({
                'date': date,
                'open': price * 0.998,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'volume': 8000000 + np.random.randint(-2000000, 2000000)
            })

        # Breakout upward (5 days)
        breakout_days = min(5, len(dates) - trend_days - consolidation_days)
        base_price = base_price
        for i, date in enumerate(dates[trend_days + consolidation_days:]):
            price = base_price * (1 + 0.03 * (i + 1))
            pattern_data.append({
                'date': date,
                'open': price * 0.995,
                'high': price * 1.008,
                'low': price * 0.995,
                'close': price,
                'volume': 20000000 + i * 2000000
            })

    elif pattern_type == 'cup_handle':
        # Create cup and handle pattern
        pattern_data = []

        # Cup formation (15 days)
        cup_days = min(15, len(dates) // 2)
        for i, date in enumerate(dates[:cup_days]):
            # Cup shape - first decline, then recovery
            if i < cup_days // 3:
                # Decline
                price = base_price * (1 - 0.15 * (i / (cup_days // 3)))
            elif i < 2 * cup_days // 3:
                # Bottom
                price = base_price * 0.85
            else:
                # Recovery
                recovery_progress = (i - 2 * cup_days // 3) / (cup_days // 3)
                price = base_price * (0.85 + 0.15 * recovery_progress)

            pattern_data.append({
                'date': date,
                'open': price * 0.998,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'volume': 10000000 + np.random.randint(-3000000, 3000000)
            })

        # Handle formation (5 days) - slight pullback
        handle_days = min(5, len(dates) - cup_days)
        base_price = base_price * 1.05  # End of cup
        for i, date in enumerate(dates[cup_days:]):
            pullback = 0.05 * np.exp(-2 * i / handle_days)  # Exponential decay
            price = base_price * (1 - pullback)
            pattern_data.append({
                'date': date,
                'open': price * 0.995,
                'high': price * 1.003,
                'low': price * 0.995,
                'close': price,
                'volume': 8000000
            })

        # Breakout (remaining days)
        breakout_days = len(dates) - cup_days - handle_days
        base_price = base_price * 0.95  # End of handle
        for i, date in enumerate(dates[cup_days + handle_days:]):
            price = base_price * (1 + 0.02 * (i + 1))
            pattern_data.append({
                'date': date,
                'open': price * 0.995,
                'high': price * 1.006,
                'low': price * 0.995,
                'close': price,
                'volume': 18000000
            })

    else:
        # Default to random walk for unknown patterns
        prices = [base_price]
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))

        pattern_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            pattern_data.append({
                'date': date,
                'open': price * 0.998,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'volume': 10000000 + np.random.randint(-4000000, 4000000)
            })

    # Convert to DataFrame
    df = pd.DataFrame(pattern_data)
    df['symbol'] = 'PATTERN'
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()

    return df.set_index('date')


def create_corrupted_data() -> pd.DataFrame:
    """
    Create corrupted data for testing data validation and error handling.

    Returns:
        DataFrame with various data quality issues
    """
    dates = pd.date_range('2023-01-01', '2023-02-28', freq='D')
    dates = dates[dates.weekday < 5]

    # Base data
    data = create_realistic_market_data('2023-01-01', '2023-02-28')

    # Introduce various data quality issues
    corrupted_data = data.copy()

    # Add missing values
    corrupted_data.loc[corrupted_data.index[10:15], 'close'] = np.nan
    corrupted_data.loc[corrupted_data.index[25:28], 'volume'] = 0

    # Add invalid OHLC relationships
    corrupted_data.loc[corrupted_data.index[30], 'high'] = corrupted_data.loc[corrupted_data.index[30], 'close'] * 0.99  # High < Close
    corrupted_data.loc[corrupted_data.index[35], 'low'] = corrupted_data.loc[corrupted_data.index[35], 'close'] * 1.01  # Low > Close
    corrupted_data.loc[corrupted_data.index[40], 'low'] = corrupted_data.loc[corrupted_data.index[40], 'high'] * 1.02  # Low > High

    # Add extreme values (outliers)
    corrupted_data.loc[corrupted_data.index[45], 'close'] = corrupted_data.loc[corrupted_data.index[45], 'close'] * 10  # 1000% price change
    corrupted_data.loc[corrupted_data.index[50], 'volume'] = 1  # Zero volume

    # Add duplicate dates
    duplicate_date = corrupted_data.index[5]
    duplicate_data = corrupted_data.loc[corrupted_data.index[4]].to_dict()
    duplicate_data['open'] = duplicate_data['open'] * 1.01
    duplicate_data['close'] = duplicate_data['close'] * 1.01
    corrupted_data = pd.concat([corrupted_data, pd.DataFrame(duplicate_data, index=[duplicate_date])])

    # Add negative prices
    corrupted_data.loc[corrupted_data.index[55], 'close'] = -50.0

    return corrupted_data


def create_large_dataset(num_rows: int = 100000) -> pd.DataFrame:
    """
    Create a large dataset for performance testing.

    Args:
        num_rows: Number of rows to generate

    Returns:
        Large DataFrame for performance testing
    """
    # Generate dates for large dataset
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=num_rows, freq='H')  # Hourly data

    # Generate prices with some patterns
    prices = [100.0]
    for i in range(1, len(dates)):
        # Add some patterns to make testing more realistic
        if i % 24 == 0:  # Daily pattern
            change = np.random.normal(0, 0.01)
        else:  # Hourly noise
            change = np.random.normal(0, 0.002)

        # Add some momentum
        if i > 1:
            change += 0.1 * (prices[-1] / prices[-2] - 1)

        prices.append(prices[-1] * (1 + change))

    # Create OHLC data
    data = pd.DataFrame({
        'open': prices[:-1],
        'close': prices[1:],
        'high': prices[:-1] * (1 + np.random.uniform(0, 0.001, len(prices)-1)),
        'low': prices[:-1] * (1 - np.random.uniform(0, 0.001, len(prices)-1)),
        'volume': np.random.uniform(1000000, 5000000, len(prices)-1)
    }, index=dates[:-1])

    # Ensure OHLC relationships
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)

    # Add technical indicators (simulate existing data)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = 100 - (100 / (1 + data['close'].pct_change().rolling(14).apply(lambda x: np.log1p(x.abs()).mean())))

    # Handle NaN values from rolling calculations
    data = data.fillna(method='bfill').fillna(method='ffill')

    return data


def create_multiple_symbols_data(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Create data for multiple symbols for concurrent testing.

    Args:
        symbols: List of symbols to create data for

    Returns:
        Dictionary mapping symbol names to DataFrames
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']

    data_dict = {}

    for symbol in symbols:
        # Create unique characteristics for each symbol
        base_multiplier = 1.0 + (hash(symbol) % 100) / 1000  # Unique price level
        volatility_factor = 0.8 + (hash(symbol) % 50) / 1000  # Unique volatility

        # Create data with unique characteristics
        data = create_realistic_market_data(symbol=symbol)

        # Apply unique scaling
        data['open'] = data['open'] * base_multiplier
        data['high'] = data['high'] * base_multiplier
        data['low'] = data['low'] * base_multiplier
        data['close'] = data['close'] * base_multiplier
        data['volume'] = data['volume'] * volatility_factor

        # Add some unique patterns
        if symbol == 'AAPL':
            # Add a V-shaped pattern
            mid_point = len(data) // 2
            data.loc[data.index[mid_point:], 'close'] = data.loc[data.index[mid_point:], 'close'] * 1.1
        elif symbol == 'TSLA':
            # Add high volatility
            price_changes = np.random.normal(0, 0.03, len(data))
            prices = [data.iloc[0]['close']]
            for change in price_changes[1:]:
                prices.append(prices[-1] * (1 + change))
            data['close'] = prices[1:]
            data['high'] = data[['high', 'open', 'close']].max(axis=1)
            data['low'] = data[['low', 'open', 'close']].min(axis=1)

        data_dict[symbol] = data

    return data_dict


def create_error_scenarios() -> Dict[str, pd.DataFrame]:
    """
    Create various error scenarios for testing robustness.

    Returns:
        Dictionary of error scenario names to problematic DataFrames
    """
    scenarios = {}

    # Empty data
    scenarios['empty'] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # Single row data
    scenarios['single_row'] = pd.DataFrame({
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100.5],
        'volume': [1000000]
    }, index=[pd.Timestamp('2023-01-01')])

    # Missing columns
    scenarios['missing_columns'] = pd.DataFrame({
        'open': [100, 101],
        'close': [100.5, 101.5],
        'volume': [1000000, 1100000]
    }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

    # All NaN data
    scenarios['all_nan'] = pd.DataFrame({
        'open': [np.nan, np.nan],
        'high': [np.nan, np.nan],
        'low': [np.nan, np.nan],
        'close': [np.nan, np.nan],
        'volume': [np.nan, np.nan]
    }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

    # Infinite values
    scenarios['infinite_values'] = pd.DataFrame({
        'open': [100, np.inf],
        'high': [101, 102],
        'low': [99, 100],
        'close': [100.5, 101.5],
        'volume': [1000000, 1100000]
    }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

    # Very large dataset (stress test)
    scenarios['large_dataset'] = create_large_dataset(10000)

    # Mixed data types
    scenarios['mixed_types'] = pd.DataFrame({
        'open': [100, '101'],
        'high': [101, 102],
        'low': [99, 100],
        'close': [100.5, 101.5],
        'volume': [1000000, 1100000]
    }, index=pd.date_range('2023-01-01', periods=2, freq='D'))

    # Future dates
    scenarios['future_dates'] = pd.DataFrame({
        'open': [100, 101],
        'high': [101, 102],
        'low': [99, 100],
        'close': [100.5, 101.5],
        'volume': [1000000, 1100000]
    }, index=pd.date_range('2050-01-01', periods=2, freq='D'))

    return scenarios


# Convenience function to get all test data
def get_all_test_data():
    """
    Get a comprehensive set of test data for all scenarios.

    Returns:
        Dictionary containing all test data scenarios
    """
    return {
        'realistic_aapl': create_realistic_market_data('AAPL'),
        'flag_pattern': create_pattern_data('flag'),
        'cup_handle_pattern': create_pattern_data('cup_handle'),
        'corrupted': create_corrupted_data(),
        'multiple_symbols': create_multiple_symbols_data(),
        'error_scenarios': create_error_scenarios(),
        'large_dataset': create_large_dataset(5000)
    }


if __name__ == "__main__":
    # Test the fixture generation
    print("Generating test data fixtures...")

    # Generate and save some sample data
    aapl_data = create_realistic_market_data('AAPL')
    print(f"Generated AAPL data: {len(aapl_data)} rows")

    pattern_data = create_pattern_data('flag')
    print(f"Generated flag pattern data: {len(pattern_data)} rows")

    corrupted_data = create_corrupted_data()
    print(f"Generated corrupted data: {len(corrupted_data)} rows")

    print("All fixtures generated successfully!")