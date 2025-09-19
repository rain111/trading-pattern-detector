# Contributing to Trading Pattern Detector

We welcome contributions! This document provides guidelines for contributing to the Trading Pattern Detector project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a positive and inclusive experience for all contributors.

## Getting Started

1. **Fork the Repository**
   - Fork the repository on GitHub
   - Clone your fork locally:
     ```bash
     git clone https://github.com/your-username/trading-pattern-detector.git
     cd trading-pattern-detector
     ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"
   ```

## Development Setup

### 1. Install Pre-commit Hooks
```bash
pre-commit install
```

### 2. Verify Setup
```bash
# Run tests
pytest

# Check code formatting
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/

# Run linting
flake8 src/ tests/
```

### 3. Project Structure
```
trading-pattern-detector/
├── src/
│   ├── trading_pattern_detector/     # Main package
│   │   ├── __init__.py
│   │   ├── cli.py                   # Command-line interface
│   │   └── core/
│   ├── detectors/                   # Pattern detectors
│   └── utils/                       # Utility functions
├── tests/                           # Test files
├── examples/                        # Usage examples
├── docs/                            # Documentation
├── .github/                         # GitHub workflows
└── pyproject.toml                   # Project configuration
```

## Submitting Changes

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific tests
pytest tests/test_detectors/
```

### 4. Format and Lint Code
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Lint code
flake8 src/ tests/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add new pattern detector"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a clear description of your changes.

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters

### Type Hints
- Use type hints for all function parameters and return values
- Import from `typing` module as needed
- Use `Optional` for nullable parameters

### Error Handling
- Use specific exception types
- Include meaningful error messages
- Log appropriate information

### Docstrings
- Use Google-style docstrings
- Include examples for public functions
- Document parameters and return values

Example:
```python
def detect_patterns(data: pd.DataFrame, symbol: str) -> List[PatternSignal]:
    """
    Detect trading patterns in market data.

    Args:
        data: OHLCV market data
        symbol: Stock symbol for analysis

    Returns:
        List of detected pattern signals

    Example:
        >>> data = pd.DataFrame({'open': [1], 'high': [2], 'low': [0.5], 'close': [1.5], 'volume': [1000]})
        >>> signals = detect_patterns(data, "AAPL")
        >>> len(signals) > 0
        True
    """
    # Implementation
```

## Testing Guidelines

### Test Structure
- Place tests in `tests/` directory
- Mirror the source directory structure
- Use descriptive test names

### Test Types
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Testing Best Practices
- Use pytest fixtures for setup
- Mock external dependencies
- Test both success and failure cases
- Aim for high test coverage

### Example Test
```python
import pytest
import pandas as pd
from trading_pattern_detector.detectors import VCPBreakoutDetector

def test_vcp_detector_initialization():
    """Test VCP detector initialization"""
    from trading_pattern_detector import PatternConfig

    config = PatternConfig(min_confidence=0.6)
    detector = VCPBreakoutDetector(config)

    assert detector.config == config
    assert hasattr(detector, 'logger')

def test_vcp_detector_basic_functionality():
    """Test basic VCP detection functionality"""
    # Create test data
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

    detector = VCPBreakoutDetector(PatternConfig())
    signals = detector.detect_pattern(data)

    assert isinstance(signals, list)
    for signal in signals:
        assert hasattr(signal, 'pattern_type')
        assert hasattr(signal, 'confidence')
```

## Documentation

### Documentation Guidelines
- Update README.md for new features
- Add docstrings to all public functions
- Include examples in documentation
- Keep API documentation current

### Building Documentation
```bash
# Install documentation dependencies
pip install -e ".[dev]"

# Build documentation (if using Sphinx)
cd docs/
make html
```

## Reporting Issues

### Bug Reports
When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Step-by-step instructions
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Minimal Reproducible Example**: Code that demonstrates the issue

### Feature Requests
For feature requests, include:

1. **Description**: Clear description of the requested feature
2. **Use Case**: Why this feature is needed
3. **Implementation Ideas**: Any thoughts on how to implement
4. **Alternatives**: Any alternatives you've considered

## Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in pyproject.toml
- Update CHANGELOG.md
- Tag releases in git

### Release Checklist
1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create git tag
6. Push to GitHub
7. Create GitHub release

## Getting Help

If you need help:
- Check the [Issues](https://github.com/tradingpatterns/trading-pattern-detector/issues) page
- Search existing issues and pull requests
- Start a new issue with detailed information
- Join our discussions in the GitHub Discussions forum

Thank you for contributing to Trading Pattern Detector!