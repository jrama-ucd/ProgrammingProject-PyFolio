# tests.py -- run with:   python3 tests.py     (Windows: python tests.py)
# Reference: Ayesha Tariq, Noaman Hakim, Muhammad Anas Asif, Justin Ramaputra, Mashhood Ahmad (Divided equally)
#
# WHAT THIS FILE IS
# -----------------
# Plain `assert` tests that PROVE the rules of the project (brief section 10 suggests
# asserts count under error handling). Each test_ function checks one rule, with a hand
# worked number in a comment, so you can re-do the arithmetic on paper at the interview.
# If every test passes the program prints "All N tests passed"; otherwise it names the
# failing test and what it expected.
#
# The tests need no data files except test_loader (it skips itself if data/training/ is
# missing) and test_storage, which writes three small files called test_*.txt in this folder
# (you can delete them afterwards).

import numpy as np

import analytics
from account import FEE, Account
from data_loader import DataNotFound, Market, load_market
from simulator import SimulationError, Simulator
from storage import Storage


def close_to(a, b):
    """Two floats are 'the same' if they differ by less than a tenth of a cent."""
    return abs(a - b) < 0.001


# A tiny 5-day market made by hand, so every expected number can be worked out on paper.
# The OPEN and CLOSE of each day are deliberately different so a wrong fill price shows.
def tiny_market():
    bars = []
    opens = [100.0, 110.0, 120.0, 130.0, 140.0]
    closes = [105.0, 115.0, 125.0, 135.0, 145.0]
    for open_price, close_price in zip(opens, closes):
        bars.append({"open": open_price, "high": close_price + 1, "low": open_price - 1,
                     "close": close_price, "volume": 1000.0})
    news = {0: ["good surge today"], 2: ["bad crash today"]}
    return Market("TINY", "", bars, news)


class RecordingAgent:
    """A fake agent that writes down EXACTLY what it was given, and buys on day 0."""

    def __init__(self):
        self.calls = []                    # one tuple per call: what decide() received

    def decide(self, bar, headlines, cash, shares, last_response):
        self.calls.append((bar, headlines, cash, shares, last_response))
        if len(self.calls) == 1:
            return ("BUY", 10)             # decide on day 0, fill on day 1
        return ("HOLD", 0)


# ----------------------------------------------------------------------- account

def test_fee_on_buy_and_sell():
    account = Account(10000.0, 0.0)
    assert account.execute(("BUY", 10), 100.0) == "ACCEPT"
    # 10 shares * 100 = 1000, fee 0.1% = 1.00, so cash = 10000 - 1001 = 8999
    assert close_to(account.cash, 8999.0) and account.shares == 10
    assert account.execute(("SELL", 10), 100.0) == "ACCEPT"
    # selling pays 1000 - 1.00 fee = 999, so cash = 8999 + 999 = 9998
    assert close_to(account.cash, 9998.0) and account.shares == 0


def test_buy_max_rounds_down_with_fee():
    # The contract's worked example: $500 at $25. Each share costs 25.025,
    # 500 / 25.025 = 19.98 -> 19 shares (NOT 20, which would cost 500.50).
    account = Account(500.0, 0.0)
    assert account.execute(("BUY", -1), 25.0) == "ACCEPT"
    assert account.shares == 19
    assert close_to(account.cash, 500.0 - 19 * 25.025)


def test_reject_cash_exact_string_and_nothing_changes():
    account = Account(100.0, 0.0)
    reply = account.execute(("BUY", 3), 50.0)       # needs 150.15, has 100
    assert reply == "REJECT currency balance: 100.0", reply
    assert account.cash == 100.0 and account.shares == 0     # a reject changes NOTHING


def test_reject_stock_exact_string_and_nothing_changes():
    account = Account(0.0, 5.0)
    reply = account.execute(("SELL", 6), 10.0)
    assert reply == "REJECT stock owned: 5.0", reply
    assert account.shares == 5.0 and account.cash == 0.0
    assert Account(1000.0, 0.0).execute(("SELL", 1), 10.0) == "REJECT stock owned: 0.0"


def test_minus_one_is_never_rejected_and_negatives_mean_max():
    assert Account(0.0, 0.0).execute(("BUY", -1), 10.0) == "ACCEPT"     # nothing to buy
    assert Account(0.0, 0.0).execute(("SELL", -1), 10.0) == "ACCEPT"    # nothing to sell
    a = Account(1000.0, 0.0)
    b = Account(1000.0, 0.0)
    a.execute(("BUY", -1), 10.0)
    b.execute(("BUY", -7), 10.0)                                        # any negative = -1
    assert a.shares == b.shares and a.cash == b.cash
    c = Account(0.0, 4.0)
    assert c.execute(("SELL", -1), 10.0) == "ACCEPT" and c.shares == 0


def test_hold_and_numpy_numbers():
    account = Account(1000.0, 0.0)
    assert account.execute(("HOLD", 0), 10.0) == "ACCEPT"
    assert account.cash == 1000.0
    assert account.execute(("BUY", np.int64(5)), 10.0) == "ACCEPT"      # numpy count is fine
    assert account.shares == 5


def test_bad_orders_and_balances_are_refused_with_errors():
    account = Account(1000.0, 0.0)
    for bad in [("JUMP", 1), ("BUY", "x"), ("BUY", float("nan")), (1, 2, 3), "BUY"]:
        try:
            account.execute(bad, 10.0)
        except ValueError:
            continue
        assert False, "order " + str(bad) + " should have been refused"
    for cash, shares in [(-500.0, 0.0), (float("nan"), 0.0), (100.0, -1.0)]:
        try:
            Account(cash, shares)
        except ValueError:
            continue
        assert False, "balances " + str((cash, shares)) + " should have been refused"


# ---------------------------------------------------------------------- simulator

def test_agent_never_sees_the_future_and_fills_at_next_open():
    sim = Simulator(tiny_market(), 10000.0, 0.0)
    agent = RecordingAgent()
    sim.run_agent(lambda: agent)                    # run_agent builds the agent by calling this
    market = tiny_market()
    # decide() is called on every day except the last: 4 calls for 5 days.
    assert len(agent.calls) == 4
    for day, call in enumerate(agent.calls):
        assert call[0] == market.bars[day]          # the bar of THAT day, nothing later
        assert call[1] == market.headlines_on(day)  # that day's headlines only
    # The first order was decided on day 0 and must fill at day 1's OPEN (110), not day 0's close.
    assert sim.trades[0][0] == 1 and sim.trades[0][2] == 110.0
    # cash = 10000 - 10 * 110 * 1.001 = 8899.0
    assert close_to(sim.account.cash, 10000.0 - 10 * 110.0 * (1 + FEE))
    # Equity is valued at the CLOSE: after day 4 -> cash + 10 * 145
    assert close_to(sim.equity[-1], sim.account.cash + 10 * 145.0)
    assert len(sim.equity) == 5                      # one value a day


def test_rejected_agent_order_reaches_the_agent_as_last_response():
    sim = Simulator(tiny_market(), 100.0, 0.0)
    agent = RecordingAgent()                         # tries to buy 10 shares with only $100
    sim.run_agent(lambda: agent)
    assert agent.calls[0][4] is None                 # first call: last_response is None
    assert agent.calls[1][4] == "REJECT currency balance: 100.0"
    assert sim.account.shares == 0 and sim.account.cash == 100.0
    assert sim.trades == []                          # a rejected order is not a trade


def test_manual_trade_fills_at_todays_close():
    sim = Simulator(tiny_market(), 10000.0, 0.0)
    assert sim.manual_trade("BUY", 10) == "ACCEPT"
    assert sim.trades[0][0] == 0 and sim.trades[0][2] == 105.0    # day 0's CLOSE, not its open
    assert close_to(sim.account.cash, 10000.0 - 10 * 105.0 * (1 + FEE))
    # today's equity is updated in place (still one entry for day 0)
    assert len(sim.equity) == 1
    assert close_to(sim.equity[0], sim.account.cash + 10 * 105.0)


def test_regimes_give_none():
    seen = []

    class Spy:
        def decide(self, bar, headlines, cash, shares, last_response):
            seen.append((bar, headlines))
            return ("HOLD", 0)

    Simulator(tiny_market()).run_agent(Spy, use_prices=False, use_news=True)
    assert all(bar is None for bar, headlines in seen)            # news-only: bar is None
    assert seen[0][1] == ["good surge today"] and seen[1][1] == []    # day 1 has no news
    seen.clear()
    Simulator(tiny_market()).run_agent(Spy, use_prices=True, use_news=False)
    assert all(headlines is None for bar, headlines in seen)      # prices-only: headlines None


def test_agent_crash_and_end_of_data_become_friendly_errors():
    class Broken:
        def decide(self, bar, headlines, cash, shares, last_response):
            raise KeyError("oops")

    try:
        Simulator(tiny_market()).run_agent(Broken)
    except SimulationError as error:
        assert "crashed on day 0" in str(error)
    else:
        assert False, "a crashing agent should raise SimulationError"
    sim = Simulator(tiny_market())
    for _ in range(4):
        sim.step()
    try:
        sim.step()
    except SimulationError:
        pass
    else:
        assert False, "stepping past the last day should raise SimulationError"


# ----------------------------------------------------------------------- analytics

def test_returns_drawdown_and_metrics_by_hand():
    equity = [100.0, 120.0, 90.0, 110.0]
    returns = analytics.daily_returns(equity)
    # 120/100-1 = 0.2 ; 90/120-1 = -0.25 ; 110/90-1 = 0.2222...
    assert close_to(returns[0], 0.2) and close_to(returns[1], -0.25)
    assert abs(returns[2] - (110 / 90 - 1)) < 1e-12
    # Running peak: 100, 120, 120, 120 -> drawdowns 0, 0, (90-120)/120 = -0.25, (110-120)/120 = -0.0833
    drawdowns = analytics.drawdown_series(equity)
    assert drawdowns[0] == 0 and drawdowns[1] == 0
    assert abs(drawdowns[2] - (-0.25)) < 1e-12 and abs(drawdowns[3] - (-10 / 120)) < 1e-12
    assert abs(analytics.max_drawdown(equity) - (-0.25)) < 1e-12
    assert abs(analytics.total_return(equity) - 0.10) < 1e-12
    expected_sharpe = np.mean(returns) / np.std(returns)
    assert abs(analytics.sharpe_like(equity) - expected_sharpe) < 1e-12
    assert analytics.sharpe_like([100.0, 100.0, 100.0]) == 0.0    # no variation: 0, not a crash


def test_moving_average_running_sum_matches_the_plain_way():
    assert list(analytics.moving_average([1, 2, 3, 4, 5], 3)) == [2.0, 3.0, 4.0]
    values = [100 + (i * 7) % 13 - i * 0.3 for i in range(60)]
    plain = [sum(values[i - 5:i]) / 5 for i in range(5, len(values) + 1)]   # slow way
    fast = analytics.moving_average(values, 5)
    assert len(fast) == len(plain)
    assert all(abs(a - b) < 1e-9 for a, b in zip(fast, plain))
    assert len(analytics.moving_average([1, 2], 5)) == 0                    # window too wide


# --------------------------------------------------------------------- data and files

def test_loader():
    try:
        market = load_market("ZNTH")
    except DataNotFound:
        print("    (skipped: no data/training/ folder next to the tests)")
        return
    assert len(market.bars) == 750 and market.news_found
    assert set(market.bars[0].keys()) == {"open", "high", "low", "close", "volume"}
    assert len(market.headlines_on(0)) >= 1
    assert market.headlines_on(10 ** 6) == []        # a day with no news is an empty list
    try:
        load_market("NO_SUCH_TICKER")
    except DataNotFound:
        pass
    else:
        assert False, "an unknown ticker should raise DataNotFound"


def test_storage_round_trip_and_leaderboard_order():
    storage = Storage("test_session.txt", "test_results.txt", "test_leaderboard.txt")
    sim = Simulator(tiny_market(), 10000.0, 0.0)
    sim.run_agent(lambda: RecordingAgent())
    storage.save_session(sim)
    data = storage.load_session()
    assert data["day"] == sim.day and data["cash"] == sim.account.cash
    assert data["equity"] == sim.equity and data["trades"] == sim.trades
    # leaderboard: write results out of order, expect best final equity first
    handle = open("test_results.txt", "w")
    handle.write("low run,9000.0,3\nhigh run,12000.5,4\nnot a valid line\nmid run,10000.0,1\n")
    handle.close()
    rows = storage.write_leaderboard()
    assert [row[1] for row in rows] == ["high run", "mid run", "low run"]
    handle = open("test_leaderboard.txt")
    lines = handle.read().splitlines()
    handle.close()
    assert lines[0] == "rank,name,final_equity" and lines[1] == "1,high run,12000.5"
    # a missing results file is an empty leaderboard, not a crash
    assert Storage("x", "no_such_results_file.txt", "test_leaderboard.txt").read_results() == []


# ------------------------------------------------------------------------- the runner

def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print("  [OK  ] " + test.__name__)
        except AssertionError as error:
            failed = failed + 1
            print("  [FAIL] " + test.__name__ + "  " + str(error))
    if failed == 0:
        print("All " + str(len(tests)) + " tests passed.")
    else:
        print(str(failed) + " of " + str(len(tests)) + " tests FAILED.")


if __name__ == "__main__":
    main()
