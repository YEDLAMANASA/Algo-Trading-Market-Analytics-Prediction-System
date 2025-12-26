# 🧩 Algo-Trading Market Analytics & Prediction System

## 📘 Project Overview
This project is a **data-driven trading analytics system** that analyzes historical stock market data, generates insights, predicts future stock movements, and simulates a simple trading strategy. It applies **data preprocessing, feature engineering, exploratory data analysis (EDA), machine learning models, and backtesting** to extract meaningful market insights and strategy performance metrics.

## 🎯 Objectives
- Collect and preprocess historical stock market data  
- Analyze price trends, trading volume, and market volatility  
- Generate technical indicators: EMA, SMA, RSI, MACD, ATR, and Volatility  
- Build ML models to predict next-day price movement: Random Forest, XGBoost, Logistic Regression, SVM  
- Simulate a Moving Average Crossover trading strategy  
- Evaluate strategy performance: profit/loss, drawdown, number of trades, win ratio  
- Generate visualizations and reports for insights  


## 🧾 Dataset Description
- Historical stock data for **RELIANCE.NS** retrieved from [Yahoo Finance]  
- Date range: **2018-01-01 to 2024-01-01**  
- Columns:  
  - `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`  

## 🧩 Workflow Steps

### 1️⃣ Data Preprocessing
- Loaded historical stock data using `yfinance`  
- Checked for missing values and duplicates  
- Converted data columns to numeric types  
- Forward filled missing values and dropped remaining NaNs  

### 2️⃣ Feature Engineering
- Generated technical indicators:  
  - EMA (20, 50), SMA (20)  
  - RSI (14), MACD (with signal and histogram)  
  - ATR (Average True Range), Volatility (20-day rolling std)  
- Created lag features for RSI, MACD, and Volatility  
- Visualized indicators for market insights  

### 3️⃣ Exploratory Data Analysis (EDA)
- Price trend analysis with moving averages  
- Volume analysis and market volatility  
- RSI and MACD behavior over time  
- Correlation heatmap of indicators  
- Candlestick chart for last 200 trading days  

### 4️⃣ Predictive Modeling
- Created target variable for next-day price movement  
- Split data into train/test and scaled features  
- Built models:  
  - Random Forest  
  - XGBoost  
  - Logistic Regression  
  - SVM  
- Compared model performance using accuracy, precision, recall, and F1-score  

### 5️⃣ Trading Strategy Simulation
- **Strategy:** Moving Average Crossover (SMA_10 & SMA_30)  
- Generated buy/sell signals  
- Calculated strategy returns, cumulative returns, drawdown  
- Evaluated performance: total profit/loss, maximum drawdown, number of trades, win ratio  
- Plotted buy/sell signals, equity curve, and drawdown curve  

## 🧠 Technologies Used
- Python 🐍  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn, XGBoost  
- mplfinance  
- Jupyter Notebook  

## 📈 Key Visualizations
| Visualization                  | Purpose                                           |
|--------------------------------|--------------------------------------------------|
| Price + Moving Averages Plot    | Show trend and crossover signals                 |
| RSI Indicator                   | Identify overbought/oversold conditions         |
| MACD Indicator                  | Track momentum and trend changes                 |
| Average True Range (ATR)        | Measure market volatility                        |
| Correlation Heatmap             | Analyze indicator relationships                  |
| Candlestick Chart               | Visualize recent market behavior                |
| MA Crossover Strategy           | Plot buy/sell signals                             |
| Equity Curve                    | Compare strategy vs market returns              |
| Drawdown Curve                  | Visualize maximum drawdowns                      |


## 🧾 Results Summary
- **Best ML Model:** Random Forest for next-day price prediction  
- **Strategy Performance:**  
  - Strategy Accuracy: ~0.71  
  - Total Profit / Loss: ~X%  
  - Maximum Drawdown: ~X%  
  - Number of Trades: ~X  
  - Win Ratio: ~X%  
- Market insights derived from technical indicators and trend analysis  

## 🧰 How to Run

## ⚙️ Installation Instructions

1. Clone the repository:
git clone https://github.com/YEDLAMANASA/Algo-Trading-Market-Analytics-Prediction-System
cd Algo-Trading-Market-Analytics-Prediction
2.Install required Python packages:
pip install -r requirements.txt
3.Open the notebook in the Jupyter Algo-Trading Market Analytics & Prediction System.ipynb
4.Run all cells sequentially to generate results and visuals.
