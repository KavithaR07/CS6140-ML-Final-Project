import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project = Path(__file__).resolve().parent.parent
price_folder = project / "data" / "raw" / "prices"

stocks = ["TSLA", "PEP", "BRK-B", "GSPC"]
prices = pd.DataFrame()

for stock in stocks:
    df = pd.read_csv(price_folder / f"{stock}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    prices[stock] = df["adj_close"]

prices = prices.loc["2010-06-29":"2026-10-30"].dropna()

normalized = prices / prices.iloc[0] * 100

plt.figure(figsize=(12, 6))

for stock in stocks:
    plt.plot(normalized.index, normalized[stock], label=stock)

plt.title("Stock Price Trends (2010–2026)")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

output = project / "results" / "figures"
output.mkdir(parents=True, exist_ok=True)

plt.savefig(output / "stock_price_trends.png")
plt.show()

print("Total trading days:", len(prices))
print("EDA graph saved successfully")

returns = prices.pct_change().dropna()

correlation = returns.corr()

plt.figure(figsize=(8, 6))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(range(len(stocks)), stocks)
plt.yticks(range(len(stocks)), stocks)

for i in range(len(stocks)):
    for j in range(len(stocks)):
        plt.text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Daily Stock Return Correlation")
plt.tight_layout()

plt.savefig(output / "stock_correlation.png", dpi=300)
plt.show()

print("Correlation heatmap saved successfully")
print(correlation)

volatility = returns.rolling(30).std() * 100

plt.figure(figsize=(12, 6))

for stock in stocks:
    plt.plot(volatility.index, volatility[stock], label=stock)

plt.title("30-Day Rolling Volatility of Stock Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return Volatility (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output / "rolling_volatility.png", dpi=300)
plt.show()

print("Rolling volatility graph saved successfully")