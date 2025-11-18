# Alpha Factor Discovery and Evaluation System

## Overview

AlphaMaster is a comprehensive quantitative trading framework for automated alpha factor discovery, optimization, and backtesting evaluation. The system integrates multiple advanced AI paradigms to generate, refine, and validate predictive financial signals across diverse market conditions.

## System Architecture

### 1. Factor Generator (`a_factor_generate`)
Multi-algorithm parallel alpha factor generation:
- **`a_agent`**: Intelligent agent-based factor generation using multi-agent systems
- **`a_gen`**: Foundational factor generation modules
- **`a_genetic`**: Evolutionary computation via genetic programming
- **`a_gfn`**: Generative flow networks for reward-aligned sampling
- **`a_miner`**: AlphaMiner-based factor discovery pipelines

### 2. Screening Optimizer (`dual_chain`)
Dual-chain architecture for systematic factor refinement:
- **Factor Evaluation**: Computation of IC, IC-IR, quintile returns, and other core metrics
- **Factor Pool Management**: Maintenance of effective and candidate factor repositories
- **Iterative Optimization**: Continuous quality enhancement through performance feedback loops

### 3. Backtest Evaluator (`compare`)
Comprehensive performance assessment framework:
- Multi-factor model construction
- Strategy backtesting execution
- Performance metric calculation
- Visual analytics and reporting

## Experimental Baselines

### Classical Factor Libraries
- **Alpha 101**: Traditional 101-factor alpha library
- **Alpha 158**: Extended 158-factor alpha universe
- **Alpha 360**: Comprehensive 360-factor collection

### Traditional Automated Methods
- **GP (Genetic Programming)**: Evolutionary expression optimization
- **DSO (Differentiable Symbolic Optimization)**: Gradient-based symbolic regression
- **AlphaGen**: Generative model-based factor discovery
- **AlphaForge**: Dynamic formulatic alpha combination framework

### LLM-Enhanced Approaches
- **LLM + CoT (Chain of Thought)**: Sequential reasoning-enhanced generation
- **LLM + ToT (Tree of Thought)**: Multi-path exploratory reasoning
- **LLM + MCTS (Monte Carlo Tree Search)**: Search-based factor optimization

## Directory Structure

```
a_factor_generate/          # Multi-paradigm factor generation
  ├── a_agent/              # Agent-based generation
  ├── a_gen/                # Foundational generators
  ├── a_genetic/            # Genetic programming
  ├── a_gfn/                # Generative flow networks
  └── a_miner/              # AlphaMiner pipelines
dual_chain/                 # Dual-chain optimization
  ├── dual_chain_manager.py # Architecture coordination
  ├── factor_evaluator.py   # Metric computation
  └── factor_pool.py        # Repository management
compare/                    # Backtesting & evaluation
  ├── alpha_evaluation.py   # Performance assessment
  └── alpha_evaluation_report.json # Analysis outputs
data/                       # Multi-market datasets
  ├── a_share/              # Chinese A-shares
  ├── crypto/               # Cryptocurrency markets
  └── us/                   # US equities
alpha_pool/                 # Factor repository storage
```

## Core Capabilities

### Factor Generation
- **Multi-paradigm Synthesis**: Integration of RL, GFlowNet, LLM, and GP methodologies
- **Diverse Factor Typologies**: Momentum, volatility, volume, statistical, and composite factors
- **Automated Expression Management**: Structured storage and retrieval of mathematical formulations

### Screening & Optimization
- **FactorEvaluator**: Computation of information coefficients, consistency metrics, and efficiency measures
- **FactorPool**: Dynamic management of factor repositories with quality thresholds
- **DualChainManager**: Coordinated exploration and exploitation workflows

### Backtesting & Evaluation
- **AlphaEvaluationSystem**: LightGBM-based multi-factor modeling
- **Comprehensive Metrics**: Returns, risk-adjusted performance, drawdown analysis
- **Automated Reporting**: Visual analytics and performance attribution

## Quick Start

### 1. Factor Generation
```bash
cd a_factor_generate/a_miner
python run_alphaminer.py
```

### 2. Factor Screening & Optimization
```bash
cd ../../dual_chain
python run_dual_chain.py
```

### 3. Backtesting & Evaluation
```bash
cd ../compare
python alpha_evaluation.py
```

## Key Performance Metrics

### Factor Quality Assessment
- **IC (Information Coefficient)**: Cross-sectional correlation between factor values and forward returns
- **IC-IR (IC Information Ratio)**: Risk-adjusted consistency measure of predictive power
- **Quintile Returns**: Performance stratification across factor-sorted portfolios

### Portfolio Performance
- **Annualized Return**: Compounded strategy performance
- **Sharpe Ratio**: Excess return per unit of volatility
- **Maximum Drawdown**: Peak-to-trough capital erosion
- **Win Rate**: Proportion of profitable periods

## Supported Markets

- **Chinese A-Shares**: CSI 500, CSI 1000 indices
- **US Equities**: S&P 500 constituents
- **Cryptocurrencies**: Major digital assets

## Technical Stack

- **Python 3.8+**: Core programming language
- **pandas, numpy**: Data manipulation and numerical computation
- **scikit-learn, LightGBM**: Machine learning and gradient boosting
- **matplotlib, seaborn**: Data visualization and analytics
- **Quantitative Libraries**: Specialized financial computation packages

## Important Notes

1. **Research Purpose**: This system is designed for academic and research applications only
2. **Computational Requirements**: Factor discovery and backtesting require substantial computational resources
3. **Environment Isolation**: Recommended deployment within virtual environments

## Development Guide

### Adding Custom Factors
1. Create new factor implementation in appropriate generator directory
2. Implement `calculate_factor` method with proper interfaces
3. Register factor in generator execution scripts

### Modifying Evaluation Metrics
1. Edit `dual_chain/factor_evaluator.py`
2. Implement new assessment methodologies
3. Update evaluation workflow logic

## Maintenance

- **Data Updates**: Regular refresh of market data to maintain factor validity
- **Parameter Calibration**: Periodic adjustment of evaluation thresholds based on regime changes
- **Algorithm Enhancement**: Continuous improvement of generation and optimization methodologies

## License

This project is intended for research purposes only. Commercial use requires explicit authorization.
