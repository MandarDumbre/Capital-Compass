
<!-- Project Title and Tagline -->
# Capital‑Compass 📈
A sleek **Streamlit & Python** application for portfolio optimization and analysis. Capital‑Compass helps users build diversified investment portfolios using modern portfolio theory, run visual analyses, and download optimized allocations.

---

<!-- Highlighting Key Features -->
## 🚀 Features

- **Efficient portfolio optimization**  
  - Calculate risk–return efficient portfolios using historical asset returns  
  - Support for mean–variance optimization & Sharpe ratio maximization

- **Interactive visualizations**  
  - Asset return distributions, correlations, and covariance matrix heatmaps  
  - Efficient frontier display with risk vs. return tradeoffs  
  - Pie charts showing portfolio weight allocations

- **Customizable inputs**  
  - Upload your own asset historical data (CSV + daily prices format)  
  - Adjust portfolio constraints (e.g., weight bounds, risk tolerance, target return)

- **One-click export**  
  - Download optimized portfolio weights and key metrics as CSV

---

<!-- Installation Instructions -->
## 📦 Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/MandarDumbre/Capital-Compass.git
   cd Capital-Compass
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch the app:

bash
Copy
Edit
streamlit run main.py
<!-- Conceptual Overview -->
🧠 How It Works
Load your data

CSV file with daily price history (dates as rows, tickers as columns)

Select assets and timeframe

Choose which assets to include in the portfolio

Pick date range to look back for historical returns

Optimization options

Define constraints (e.g. weight limits, target returns)

Set goal: minimum variance vs. optimum Sharpe ratio

Analyze & Export

Visualize portfolio efficient frontier and allocation breakdowns

Download optimized portfolio weights and performance metrics

<!-- Optional: Add visuals or screen recordings of your app -->
📈 Screenshots
<!-- Replace the links below with actual image links -->


<!-- User Journey or Usage Steps -->
💡 Typical Workflow
Provide cleaned daily price CSV files

Choose a timeframe (e.g., past 3 years)

Set constraints (0–100% per asset; full allocation to 100%)

Run optimization — view result, weights, risk/return metrics

Export the portfolio for real‑world implementation

<!-- Guide for adding new features or forking -->
🛠️ Customization & Extension
Swap in alternate optimization methods (e.g., Black‑Litterman, CVaR)

Add support for live price feeds or multiple asset classes

Extend user interface with reporting or scenario‑analysis features

Change weight bounds to allow short positions or leverage

<!-- List of Python packages or dependencies -->
📋 Requirements
Python 3.8+

pandas

numpy

cvxpy

Streamlit

matplotlib or plotly

<!-- Academic or theoretical background -->
📚 References
Markowitz Mean–Variance Theory

Portfolio optimization metrics:

Expected Returns

Standard Deviation (Volatility)

Sharpe Ratio

Correlation & Covariance

<!-- Contribution Guidelines -->
🤝 Contributing
Fork the repo

Create a feature branch (git checkout -b new-feature)

Add code, tests, and documentation

Open a Pull Request — feedback is welcome!

<!-- How users can reach you or open issues -->
📬 Contact
For questions, feature requests, or contributions, feel free to open an issue
or contact Mandar Dumbre at [dumbremandar@gmail.com].

<!-- Licensing Info -->
🧭 License
Distributed under the MIT License.
Use freely in personal or commercial projects.
