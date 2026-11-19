# analytics.py
# Reference: Ayesha Tariq, Mashhood Ahmad
#
# WHAT THIS FILE IS
# -----------------
# All the numbers and pictures about a run:
#   * indicators built BY HAND (no pandas time-series shortcuts, as the library policy
#     requires): daily returns, a moving average;
#   * the three metrics: total return, max drawdown, sharpe-like score;
#   * the three charts: price with trades, equity curve, drawdown.
# The recipes are the ones in project document 09 (the time-series supplement).
#
# Everything here works on plain lists / numpy arrays and knows nothing about the
# Account, the menu or files. Libraries: numpy for slicing and averages, matplotlib
# (pyplot) for charts. That is the whole toolbox.

import numpy as np

# matplotlib is only needed for the charts. If it is missing we do not crash at start-up;
# the chart functions say so politely when somebody asks for a chart.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# The moving-average window drawn on the price chart (days).
MA_WINDOW = 10


class AnalyticsError(Exception):
    """Raised when a metric or chart cannot be produced; the message is for the user."""


# ----------------------------------------------------------------------------------
# Indicators, built by hand
# ----------------------------------------------------------------------------------

def daily_returns(values):
    """Day-to-day return of a list of values: today / yesterday - 1.

    Example: [100, 120, 90] -> [0.2, -0.25].
    Two slices of the same array, one day apart: values[1:] is "today",
    values[:-1] is "yesterday". Element i of each is the same pair of days.
    O(n): one pass, done by numpy.
    """
    values = np.array(values, dtype=float)       # an ARRAY before slicing (the trap in doc 09)
    if len(values) < 2:
        return np.array([])
    today = values[1:]
    yesterday = values[:-1]
    # A yesterday of 0 would divide by zero. That can only happen with an empty account;
    # we answer 0 return for such a day instead of crashing or producing inf/NaN.
    safe_yesterday = np.where(yesterday > 0, yesterday, 1.0)
    returns = today / safe_yesterday - 1
    return np.where(yesterday > 0, returns, 0.0)


def moving_average(values, window):
    """Simple moving average: the mean of the last `window` values, moved forward a day at a time.

    Returns an array of length len(values) - window + 1. Element 0 is the mean of
    values[0:window], so it belongs to day window-1 (the first day with enough history).

    THE OPTIMIZATION (named in the report's complexity note):
    The obvious way re-adds `window` numbers for every day: O(n * window).
    Instead we keep a RUNNING SUM: when the window moves one day forward we add the
    new value and subtract the one that just left. That is 2 operations per day no
    matter how wide the window is, so the whole function is O(n).
    """
    values = np.array(values, dtype=float)
    if window < 1:
        raise AnalyticsError("the moving-average window must be at least 1 day")
    if len(values) < window:
        return np.array([])
    running_sum = float(values[:window].sum())   # the first window, added once
    averages = [running_sum / window]
    for i in range(window, len(values)):
        running_sum = running_sum + values[i] - values[i - window]
        averages.append(running_sum / window)
    return np.array(averages)


# ----------------------------------------------------------------------------------
# The three metrics
# ----------------------------------------------------------------------------------

def total_return(equity):
    """(final equity - starting equity) / starting equity. 0.1 means +10%."""
    if len(equity) < 1 or equity[0] <= 0:
        return 0.0
    return (equity[-1] - equity[0]) / equity[0]


def drawdown_series(equity):
    """For every day, how far equity sits below its highest value so far, as a fraction.

    0.0 means 'at a new high'; -0.25 means 'down 25% from the best so far'.
    Running-peak loop from project document 09. O(n): one pass.
    """
    drawdowns = []
    if len(equity) > 0:
        peak = equity[0]
    else:
        peak = 0.0
    for value in equity:
        if value > peak:
            peak = value                      # a new best value so far
        if peak > 0:
            drawdowns.append((value - peak) / peak)
        else:
            drawdowns.append(0.0)
    return drawdowns


def max_drawdown(equity):
    """The worst (most negative) value of the drawdown series. -0.29 = a 29% fall."""
    drawdowns = drawdown_series(equity)
    if len(drawdowns) == 0:
        return 0.0
    return min(drawdowns)


def sharpe_like(equity):
    """mean(daily returns of the equity) / standard deviation of those returns.

    A simple risk-adjusted score (the data has no dates, so no real Sharpe ratio).
    np.std divides by n (not n-1), the same as the competition's scorer, so our number
    matches theirs. If the returns never vary (std = 0, e.g. an agent that never
    trades) the ratio is undefined; we report 0.0 instead of crashing.
    """
    returns = daily_returns(equity)
    if len(returns) == 0:
        return 0.0
    spread = float(np.std(returns))
    if spread == 0.0:
        return 0.0
    return float(np.mean(returns)) / spread


class RunStats:
    """The numbers of one finished (or in-progress) run, ready to print and to save."""

    def __init__(self, name, equity, trade_count):
        self.name = name
        self.final_equity = float(equity[-1])
        self.total_return = total_return(equity)
        self.max_drawdown = max_drawdown(equity)        # negative, e.g. -0.291
        self.sharpe = sharpe_like(equity)
        self.trade_count = trade_count

    def __str__(self):
        """The block of figures the desk prints after a run (week 7: __str__)."""
        return ("  " + self.name + "\n"
                + "    final equity : " + str(round(self.final_equity, 2)) + "\n"
                + "    total return : " + str(round(self.total_return * 100, 1)) + "%\n"
                + "    sharpe-like  : " + str(round(self.sharpe, 3)) + "\n"
                + "    max drawdown : " + str(round(abs(self.max_drawdown) * 100, 1)) + "%\n"
                + "    trades       : " + str(self.trade_count))


# ----------------------------------------------------------------------------------
# The three charts. Each one is drawn, saved as a PNG next to the program (handy for the
# report) and shown on screen. plt.show() WAITS until you close the window.
# ----------------------------------------------------------------------------------

def _need_matplotlib():
    if plt is None:
        raise AnalyticsError("matplotlib is not installed, so charts are unavailable. "
                             "Install it with: python3 -m pip install matplotlib")


def _finish(filename):
    """Save the current figure, show it, then close it so figures do not pile up."""
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_price_with_trades(ticker, bars, trades, filename="price_with_trades.png"):
    """Chart 1: the closing price, the moving average, and every trade as a triangle.

    trades is a list of (day, side, price, shares), where `day` is the day the order
    FILLED and `price` is the price it filled at (so a triangle always sits on the line
    for a manual trade at the close, and on the day's open for an agent trade).
    """
    _need_matplotlib()
    closes = np.array([bar["close"] for bar in bars])        # array before use
    plt.figure(figsize=(10, 5))
    plt.plot(closes, color="gray", label="close")             # gray keeps the price out of the way
    averages = moving_average(closes, MA_WINDOW)
    if len(averages) > 0:
        # averages[0] belongs to day MA_WINDOW-1, so we give plot the matching day numbers.
        days_for_average = list(range(MA_WINDOW - 1, len(closes)))
        plt.plot(days_for_average, averages, color="blue", linewidth=1,
                 label=str(MA_WINDOW) + "-day average")
    # Two plain lists per side: the 'accumulator' habit from week 1.
    buy_days = [t[0] for t in trades if t[1] == "BUY"]
    buy_prices = [t[2] for t in trades if t[1] == "BUY"]
    sell_days = [t[0] for t in trades if t[1] == "SELL"]
    sell_prices = [t[2] for t in trades if t[1] == "SELL"]
    plt.scatter(buy_days, buy_prices, marker="^", color="green", label="buy", zorder=3)
    plt.scatter(sell_days, sell_prices, marker="v", color="red", label="sell", zorder=3)
    plt.title(ticker + ": price with trades")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    _finish(filename)


def plot_equity(ticker, equity, filename="equity_curve.png"):
    """Chart 2: equity (cash + shares * close) for every day so far."""
    _need_matplotlib()
    plt.figure(figsize=(10, 5))
    plt.plot(equity, color="darkblue")
    plt.axhline(equity[0], color="gray", linestyle="--", linewidth=1)   # the starting value
    plt.title(ticker + ": equity curve")
    plt.xlabel("Day")
    plt.ylabel("Equity (USD)")
    _finish(filename)


def plot_drawdown(ticker, equity, filename="drawdown.png"):
    """Chart 3: how far equity sits below its best value so far (0 = at the top)."""
    _need_matplotlib()
    plt.figure(figsize=(10, 5))
    plt.plot(drawdown_series(equity), color="firebrick")
    plt.title(ticker + ": drawdown")
    plt.xlabel("Day")
    plt.ylabel("Drawdown (fraction below best so far)")
    _finish(filename)
