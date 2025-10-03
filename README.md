# Systematic Commodity Return Forecasting Engine

## Project Summary

This project is a sophisticated, two-stage modeling framework designed to generate daily return forecasts for 424 commodity-related assets from the MITSUI&CO. Kaggle competition. The architecture is built to first model and filter for systemic market risk before generating asset-specific alpha signals, creating a robust system for navigating different market volatility regimes.

## Technical Highlights

* **Two-Stage Hierarchical Modeling:** A global market regime model's output is used as a predictive feature for a large set of specialized, asset-specific models. This decouples the forecast of systemic risk from idiosyncratic asset behavior.
* **Custom Volatility Feature:** An early-warning signal for market turbulence was engineered by modeling the "volatility of volatility" of a major US equity index, capturing shifts in market risk appetite.
* **Massively Parallel Model Training:** The framework trains and manages 424 distinct `LGBMRegressor` models, one for each target asset, allowing the system to learn the unique dynamics of each instrument.
* **Advanced Feature Engineering:** The feature set for each model is rich and multi-faceted, incorporating autoregressive (momentum), cross-asset (market trend), and risk-regime signals.

## Modeling Architecture

The core of the strategy is a pipeline that processes market data through two distinct modeling components.

#### Component 1: Market Regime Filter

A primary model is trained to forecast forward-looking market volatility. This is a crucial first step, as asset correlations and behaviors change dramatically between low- and high-volatility environments.

* **Objective:** Classify the market state 5 days forward, identifying periods of high tail risk.
* **Model:** `LGBMClassifier`.
* **Primary Signal:** The model's core input is a feature derived from the term structure of volatility of a market proxy, designed to act as a leading indicator of systemic risk.
* **Output:** A daily probability of a high-volatility regime.

#### Component 2: Asset-Specific Alpha Models

A dedicated `LGBMRegressor` is trained for each of the 424 target assets. This approach was chosen over a single monolithic model to better capture the unique characteristics and dynamics of different asset classes (e.g., Metals, Futures, FX).

The feature set for each of the 424 models includes:

* **Market Regime Signal:** The lagged probability output from the Component 1 model, conditioning each forecast on the expected level of systemic risk.
* **Autoregressive Features:** The target's own historical returns (Lags 1, 2, and 3) to model short-term momentum.
* **Cross-Asset Features:** Rolling statistical measures (mean/std over 5- and 21-day windows) of key global instruments to capture broader market trends and inter-market relationships.
* **Primary Market Data:** Raw price/volume information specific to the asset.

## Tech Stack

* **Core Libraries:** Python, Pandas, NumPy, Scikit-learn
* **Modeling:** LightGBM
* **Data Handling:** Polars (for inference optimization in the Kaggle API)
