<div align="center">

# 🧭 **Capital Compass**

### *Navigate Your Investments with Data-Driven Intelligence*

**An advanced portfolio optimization tool applying Modern Portfolio Theory with live market data and AI-powered insights.**

</div>

---

## 🎥 Capital Compass Demo

<a href="https://drive.google.com/file/d/1MWKVONO_JM8jras-fuV9z9Fkzo2eVLEB/view?usp=drive_link" target="_blank">
<img src="https://drive.google.com/uc?export=view&id=1xjNJw_X_fwWPdYlvsJBdWDBCLeE17dIa" alt="Capital Compass Dashboard Preview" width="920"/>
</a>

<p align="center">
  <a href="https://drive.google.com/file/d/1MWKVONO_JM8jras-fuV9z9Fkzo2eVLEB/view?usp=drive_link" target="_blank">
    ▶️ <strong>Watch the Full Video Demo</strong>
  </a>
</p>

---

<div align="center">

🛠️ Built With  
<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white" />
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas&logoColor=white" />
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-Scientific_Computing-013243?logo=numpy&logoColor=white" />
<img alt="Plotly" src="https://img.shields.io/badge/Plotly-Interactive_Charts-3f4f75?logo=plotly&logoColor=white" />
<img alt="PyPortfolioOpt" src="https://img.shields.io/badge/PyPortfolioOpt-Optimization-8A2BE2?logo=python&logoColor=white" />
<img alt="yfinance" src="https://img.shields.io/badge/yfinance-Market_Data-008080?logo=yahoo&logoColor=white" />
<img alt="SQLite" src="https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white" />
<img alt="Llama 3" src="https://img.shields.io/badge/Llama_3-AI_Insights-E91E63?logo=meta&logoColor=white" />
</p>
</div>

---

## 🚀 Features

| 🔧 Feature                  | ⚡ Description                                                                                               |
| ---------------------------| ----------------------------------------------------------------------------------------------------------- |
| 📊 **MPT Optimization**     | Generates three optimal portfolios (Low, Medium, High Risk) based on Modern Portfolio Theory.               |
| 📈 **Advanced Risk Metrics**| Calculates Sortino Ratio, Max Drawdown, Calmar Ratio, and Value at Risk (VaR) for deep risk analysis.       |
| 🎲 **Monte Carlo Simulation**| Forecasts future performance using thousands of simulations to show a range of outcomes.                   |
| 💡 **AI-Powered Insights** | Leverages **Llama 3** via Groq to generate qualitative summaries of portfolio strategies.                   |
| 🎨 **Interactive Charts**  | Visualizes the **Efficient Frontier**, normalized performance, and Monte Carlo results.                      |
| 💰 **Discrete Allocation** | Converts percentages into real share quantities based on your investment amount.                            |
| ⚡ **Data Caching**        | Uses **SQLite** to cache historical stock data, ensuring fast load and fewer API calls.                     |
| 🌍 **Multi-Market Support**| Includes US and Indian stocks out of the box.                                                              |

---

## 🛠️ Technology Stack

```text
Application Framework : Streamlit  
Core Logic            : Python, PyPortfolioOpt  
Data Handling         : Pandas, NumPy  
Data Fetching         : yfinance  
Database              : SQLite  
Visualization         : Plotly  
AI Integration        : Groq API (Llama 3 8B)
````

---

## 📋 Prerequisites

* Python 3.9+
* pip (Python package manager)
* A free Groq API Key for AI Insights

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/capital-compass.git
cd capital-compass

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

---

## 🧠 API Key Setup

This project uses the Groq API to generate AI insights.

1. Get a free API key from [Groq Console](https://console.groq.com/keys)
2. Create a file named `.env` in the project root directory
3. Add this line inside the `.env` file:

```bash
GROQ_API_KEY="your-api-key-here"
```

---

## ▶️ Run the App

```bash
streamlit run main.py
```

---

## 📊 Financial Metrics Overview

| 🔍 Metric               | 📌 Purpose                                        |
| ----------------------- | ------------------------------------------------- |
| **Sharpe Ratio**        | Measures risk-adjusted return. Higher is better.  |
| **Sortino Ratio**       | Like Sharpe, but focuses only on downside risk.   |
| **Max Drawdown**        | The largest peak-to-trough drop in the portfolio. |
| **Calmar Ratio**        | Measures return relative to the Max Drawdown.     |
| **Value at Risk (VaR)** | Potential daily loss with 95% confidence level.   |

---

## 🎯 Usage Guide

1. **🚀 Launch the application**
2. **💼 Enter your investment amount** in the sidebar.
3. **📈 Select 5–10 stocks** from the list.
4. **🖱️ Click "Generate Portfolio Analysis"**
5. **🔎 Explore the outputs:**

   * Low, Medium, and High-Risk portfolios with allocation breakdown
   * AI-generated summary of portfolios
   * Interactive visualizations

---

## 🗂️ Project Structure

```bash
capital-compass/
├── main.py                    # Main Streamlit application
├── requirements.txt           # Project dependencies
├── .env                       # API Key (you create this)
├── README.md                  # This file
└── src/
    ├── config.py              # Static config data
    ├── __init__.py
    ├── data/
    │   ├── database.py        # DB handling logic
    │   └── stock_data.db      # SQLite database
    ├── logic/
    │   ├── analysis.py        # Fetching + AI logic
    │   ├── metrics.py         # Custom financial metrics
    │   └── optimizer.py       # Optimization engine
    └── ui/
        └── plots.py           # All visualization components
```

---

## 🛡️ Security & Privacy

✔️ All processing is **local** — no user data is stored online
✔️ Stock data via **yfinance** — real-time, reliable API
✔️ API Key only used for secure Groq requests — stored in `.env`

---

## 🔮 Roadmap

* [ ] 🧪 **Backtesting Engine**
* [ ] 💹 **ETFs, Crypto, and Bonds**
* [ ] 👤 **User Accounts & Tracking**
* [ ] 🧠 **Factor-Based Optimization**
* [ ] 📄 **Exportable PDF Reports**

---

## 📝 License

This project is licensed under the **MIT License** — see `LICENSE` for details.

---

## 🙏 Acknowledgments

Big thanks to the creators of these amazing tools:

* [Streamlit](https://streamlit.io/)
* [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)
* [Plotly](https://plotly.com/)
* [Pandas](https://pandas.pydata.org/)
* [yfinance](https://github.com/ranaroussi/yfinance)

---

<div align="center">

⭐ **Star this repo if you find it useful**
🐛 [Report a Bug](https://github.com/your-username/capital-compass/issues) • ✨ [Request a Feature](https://github.com/your-username/capital-compass/issues)

</div>

