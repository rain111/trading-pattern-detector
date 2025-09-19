 #!/usr/bin/env python3
"""
Research and compare backtesting libraries: vectorbt vs backtrader vs zipline vs empyrial
"""

import subprocess
import sys
import importlib.util
import time

def check_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def test_vectorbt():
    """Test vectorbt library"""
    print("🔍 Testing vectorbt...")

    try:
        import vectorbt as vbt

        # Create sample data
        price = pd.Series([100, 101, 102, 103, 102, 101, 100, 99, 98, 99])

        # Create portfolio
        pf = vbt.Portfolio.from_orders(
            price=price,
            size=[1, 0, -1, 0, 1, 0, -1, 0, 1, 0],
            fees=0.001
        )

        print(f"✅ vectorbt works - Final PnL: {pf.total_pnl()}")
        return {
            'library': 'vectorbt',
            'version': vbt.__version__,
            'pros': [
                'Very fast (vectorized operations)',
                'Built on pandas/numpy',
                'Excellent plotting capabilities',
                'Multi-asset support',
                'Good documentation'
            ],
            'cons': [
                'Steep learning curve',
                'Higher memory usage',
                'Complex for simple strategies'
            ]
        }
    except Exception as e:
        print(f"❌ vectorbt failed: {e}")
        return None

def test_backtrader():
    """Test backtrader library"""
    print("🔍 Testing backtrader...")

    try:
        import backtrader as bt

        # Create simple strategy
        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.dataclose = self.datas[0].close
                self.order = None

            def next(self):
                if not self.position:
                    self.order = self.buy()
                else:
                    self.order = self.sell()

        # Create cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(TestStrategy)

        # Add data
        data = bt.feeds.PandasData(dataname=pd.DataFrame({
            'open': [100, 101, 102, 103, 102],
            'high': [101, 102, 103, 104, 103],
            'low': [99, 100, 101, 102, 101],
            'close': [100.5, 101.5, 102.5, 103.5, 102.5],
            'volume': [1000, 1100, 1200, 1300, 1200]
        }))
        cerebro.adddata(data)

        # Add sizer
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

        print("✅ backtrader works - Strategy creation successful")
        return {
            'library': 'backtrader',
            'version': bt.__version__,
            'pros': [
                'Event-driven architecture',
                'Flexible and extensible',
                'Good for complex strategies',
                'Built-in analyzers',
                'Active community'
            ],
            'cons': [
                'Slower for large datasets',
                'More verbose code',
                'Limited built-in indicators'
            ]
        }
    except Exception as e:
        print(f"❌ backtrader failed: {e}")
        return None

def test_empyrical():
    """Test empyrical library"""
    print("🔍 Testing empyrical...")

    try:
        import empyrical

        # Sample returns
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

        # Calculate metrics
        sharpe = empyrical.sharpe_ratio(returns)
        max_drawdown = empyrical.max_drawdown(returns)

        print(f"✅ empyrical works - Sharpe: {sharpe:.3f}, Max Drawdown: {max_drawdown:.3f}")
        return {
            'library': 'empyrical',
            'version': empyrical.__version__,
            'pros': [
                'Excellent for risk metrics',
                'Comprehensive performance analysis',
                'Industry-standard calculations'
            ],
            'cons': [
                'Limited to metrics only',
                'No backtesting engine',
                'Requires separate portfolio management'
            ]
        }
    except Exception as e:
        print(f"❌ empyrical failed: {e}")
        return None

def main():
    """Main function to test all libraries"""
    print("🔍 Researching Backtesting Libraries")
    print("=" * 50)

    results = []

    # Test vectorbt
    if check_package_installed('vectorbt'):
        result = test_vectorbt()
    else:
        print("📦 Installing vectorbt...")
        if install_package('vectorbt'):
            result = test_vectorbt()
        else:
            print("❌ Failed to install vectorbt")
            result = None

    if result:
        results.append(result)

    # Test backtrader
    if check_package_installed('backtrader'):
        result = test_backtrader()
    else:
        print("📦 Installing backtrader...")
        if install_package('backtrader'):
            result = test_backtrader()
        else:
            print("❌ Failed to install backtrader")
            result = None

    if result:
        results.append(result)

    # Test empyrical
    if check_package_installed('empyrical'):
        result = test_empyrical()
    else:
        print("📦 Installing empyrical...")
        if install_package('empyrical'):
            result = test_empyrical()
        else:
            print("❌ Failed to install empyrical")
            result = None

    if result:
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("📊 LIBRARY COMPARISON SUMMARY")
    print("=" * 50)

    for result in results:
        print(f"\n📚 {result['library']} v{result['version']}")
        print(f"   ✅ Pros: {', '.join(result['pros'][:3])}")
        print(f"   ❌ Cons: {', '.join(result['cons'][:3])}")

    # Recommendation
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATION")
    print("=" * 50)

    if results:
        # vectorbt is recommended for this use case
        if 'vectorbt' in [r['library'] for r in results]:
            print("🏆 RECOMMENDED: vectorbt")
            print("   - Best for performance with large datasets")
            print("   - Excellent for pattern-based strategies")
            print("   - Great visualization capabilities")
            print("   - Can be combined with empyrical for metrics")

        elif 'backtrader' in [r['library'] for r in results]:
            print("🏆 RECOMMENDED: backtrader")
            print("   - Good for complex event-driven strategies")
            print("   - More traditional backtesting approach")
            print("   - Good for learning purposes")

        print("\n💡 Strategy: Use vectorbt + empyrical combination")
        print("   - vectorbt for backtesting engine")
        print("   - empyrical for advanced risk metrics")

    return results

if __name__ == "__main__":
    # Import pandas (required by some libraries)
    import pandas as pd
    results = main()