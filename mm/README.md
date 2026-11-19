# Market Making Hackathon

Ten stocks, ten rounds. Each round you predict the price of one hidden stock from data, then quote a **bid** and an **ask**. The team with the tightest spread becomes the **market maker** and must accept every trade at its prices; every other team trades against it. The true price is revealed, everyone is paid out, and the team with the most money after ten rounds wins.

The paritcipation website is: https://marketmaking-hackathon.onrender.com/admin

**Read the full rules first:** [`instructions/participant_guide.pdf`](instructions/participant_guide.pdf)

## What's in this repository

| Path | Contents |
|------|----------|
| [`instructions/`](instructions/) | The participant guide (PDF and LaTeX source): rules, a worked example, scoring and strategy. |
| [`notebooks/01_getting_started_stock_1.ipynb`](notebooks/01_getting_started_stock_1.ipynb) | A beginner-friendly, step-by-step notebook: load the data, look at it, fit a linear regression and turn its prediction into a quote for stock 1. No prior experience needed. |
| [`datasets/`](datasets/) | The data. |
| `requirements.txt` | Python libraries used by the notebook. |

## The data

For each stock `k` (1 to 10) there are two files in `datasets/`:

| File | Contents |
|------|----------|
| `stock_<k>_train.csv` | Historical observations: feature columns plus the target column `price` (in £). |
| `stock_<k>_test.csv`  | **One row** with the same feature columns and no `price`. This is the stock you quote in round *k*. |

The `price` of each test row is known only to the organisers and is revealed when its round settles.

- Every row has a unique `uid`. Test files also have an `id` column naming the stock.
- Column names are anonymised. Missing values are empty cells.
- The datasets differ in size, number of features, noise and difficulty. Do not assume that what worked on one stock works on the next. Everything you need is in this repository, so look at it.

## Each round in brief

1. **Quote.** Submit one bid and one ask (to the penny; the bid must be above £0 and below the ask). Quotes are sealed (no one else sees yours until the round settles, unless you become market maker) and final once submitted.
2. **Market maker.** The smallest spread (ask − bid) wins. If spreads tie exactly, the quote that reached the server **first** wins. Only that team's quote becomes the market, so tied quotes are never combined.
3. **Trade.** Every other team makes one trade with the market maker: buy at its ask or sell at its bid, at least 10 shares, worth at most your cash (capped at £250,000). Teams that don't trade get a random 10-share trade.
4. **Settle.** The true price is revealed and every position is paid out at it.

You start with £100,000 and are ranked by portfolio value. The guide has the details.

## AI tools

AI assistants are allowed. We strongly recommend that you understand what your code does: once the competition is live you'll need to make quick decisions about how wide to quote and when to trade, and that is much easier when you know where your numbers come from and when to doubt them.

## Getting started

```bash
git clone <this repository>
cd <repository folder>
pip install -r requirements.txt        # Python 3.10 or newer
jupyter lab notebooks/01_getting_started_stock_1.ipynb
```

You can also open the notebook in VS Code, or upload it with the `datasets/` folder to Google Colab.

Good luck, and look at your data.
