# Quantum Decision Theory

A fair comparison framework demonstrating genuine quantum advantages in decision-making models through quantum phenomena rather than inflated parameters.

## Overview

This project implements and compares Quantum Decision Theory (QDT) with Classical Decision Theory using calibrated parameters to ensure fair comparison. The quantum advantages come from genuine quantum phenomena like:

- **Order Effects**: Information presentation sequence affects decisions
- **Quantum Interference**: Context factors create interference patterns  
- **Superposition**: Decision uncertainty until measurement
- **Entanglement**: Correlated decision factors

## Key Features

- **Fair Parameter Calibration**: Both quantum and classical models use identical maximum effect sizes (15% for major factors)
- **Interactive Streamlit Demo**: Visual comparison of model behaviors
- **Scientific Validity**: Testable predictions based on real quantum phenomena
- **Business Applications**: Consumer behavior modeling with quantum effects

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Usage

### Run the Interactive Demo

```bash
# Run with uv
uv run streamlit run fair_qdt_demo_app.py
```

The demo allows you to:
- Adjust customer profiles (price sensitivity, quality preference, brand loyalty)
- Configure store context (lighting, music, crowding, display quality)  
- Test different information presentation orders
- Compare quantum vs classical predictions

### Run Command Line Comparison

```bash
# Run with uv
uv run python fair_quantum_decision_theory.py
```

## Project Structure

- `fair_quantum_decision_theory.py` - Core quantum and classical decision models
- `fair_qdt_demo_app.py` - Interactive Streamlit visualization app
- `pyproject.toml` - Project dependencies and configuration

## Dependencies

- **Quantum Computing**: Qiskit, Qiskit-Aer, Qiskit-Machine-Learning
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Plotly, Streamlit

## Model Comparison

### Classical Model
- Order independent (information sequence doesn't matter)
- Linear utility combination
- Predictable interactions between factors

### Quantum Model  
- Order dependent (violates classical probability)
- Quantum entanglement between decision factors
- Interference patterns in context effects
- Superposition collapse upon measurement

## Fair Comparison Principles

1. **Same Parameter Ranges**: Both models use identical maximum effect sizes
2. **No Artificial Inflation**: Quantum effects are not artificially amplified
3. **Genuine Phenomena**: Advantages come from real quantum behaviors
4. **Testable Predictions**: Order effects and interference patterns are experimentally verifiable

## Business Applications

- **Retail Strategy**: Optimize information presentation sequence
- **Marketing**: Understand context-dependent consumer behavior
- **Product Placement**: Leverage quantum interference effects
- **Customer Experience**: Design environments considering quantum decision effects

## Scientific Background

The implementation is based on quantum decision theory research showing that human decision-making can exhibit quantum-like properties including:
- Violation of classical probability rules
- Context-dependent preferences
- Order effects in information processing
- Uncertainty and superposition in decision states

## License

MIT