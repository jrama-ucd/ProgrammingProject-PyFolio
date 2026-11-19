# simulator.py
# Reference: Noaman Hakim
#
# WHAT THIS FILE IS
# -----------------
# The Simulator is the "engine": it walks through the market day by day, asks the agent
# for an order, hands the order to the Account, and writes down the equity.
#
# THE TWO FILL RULES (the heart of the project, brief section 10):
#
#   * AGENT orders (the back-test): the agent decides at the END of day t, seeing days
#     0..t only. The order fills at the OPEN of day t+1. The agent is never given the
#     future, so look-ahead is impossible.
#   * MANUAL orders (you typing buy/sell): you are standing at the counter and the price
#     is on the board, so the trade fills at TODAY'S CLOSE.
#
# Equity is recorded once per day as cash + shares * that day's CLOSE.
#
# Separation of concerns (rubric item 2):
#     TradingAgent (agent.py)  only THINKS: it returns a request.
#     Account (account.py)     is the AUTHORITY: it holds the money and can refuse.
#     Simulator (this file)    connects the two and keeps the calendar.
# The Simulator never decides what to trade, and the agent never touches the money.

import random

import numpy as np

from account import Account, is_number


class SimulationError(Exception):
    """Raised for anything that stops a run (e.g. the agent crashed). Message is for the user."""


class Simulator:
    """One trading session on one market: a calendar, an Account and the history so far."""

    def __init__(self, market, cash=10000.0, shares=0.0):
        self.market = market
        self.reset(cash, shares)

    # ------------------------------------------------------------------ set-up

    def reset(self, cash, shares):
        """Go back to day 0 with fresh balances (used on load and when balances are set)."""
        self.start_cash = float(cash)
        self.start_shares = float(shares)
        self.account = Account(cash, shares)      # raises ValueError for daft balances
        self.day = 0
        # equity[d] is the value at the close of day d, so equity is always day+1 long.
        self.equity = [self.account.value(self.market.bars[0]["close"])]
        # Every fill is logged as (day, "BUY"/"SELL", price, shares) for the price chart.
        self.trades = []
        self.last_order = "none yet"              # the latest event: the order...
        self.last_reply = ""                      # ...and the desk's reply to it
        self.last_response = None                 # what the agent gets as last_response
        self.agent_name = ""
        self.regime = ""                          # what the agent saw: both / prices only / news only
        self.run_finished = False                 # True once an agent has run to the end
        self.run_saved = False                    # True once that run is in the results file

    def restore(self, day, cash, shares, equity, trades, start_cash, start_shares,
                agent_name="", regime="", run_finished=False, run_saved=False,
                last_order="none yet", last_reply=""):
        """Put a saved session back (called by the desk after Storage reads the file)."""
        if not (0 <= day < len(self.market.bars)):
            raise SimulationError("the saved day " + str(day) + " does not fit this market "
                                  "(it has " + str(len(self.market.bars)) + " days)")
        if len(equity) != day + 1:
            raise SimulationError("the saved equity list does not match the saved day")
        self.account = Account(cash, shares)
        self.start_cash = float(start_cash)
        self.start_shares = float(start_shares)
        self.day = day
        self.equity = list(equity)
        self.trades = list(trades)
        self.agent_name = agent_name
        self.regime = regime
        self.run_finished = run_finished
        self.run_saved = run_saved
        self.last_order = last_order              # the latest event survives a save / load
        self.last_reply = last_reply

    # --------------------------------------------------------------- read-only helpers

    def last_day(self):
        """Index of the final day (a market of 750 days ends on day 749)."""
        return len(self.market.bars) - 1

    def close_today(self):
        return self.market.bars[self.day]["close"]

    def at_end(self):
        return self.day >= self.last_day()

    def status_line(self):
        """The line the desk prints after every action: day, price, balances, equity."""
        return ("day " + str(self.day) + "/" + str(self.last_day())
                + " | price " + str(round(self.close_today(), 2))
                + " | wallet " + str(round(self.account.cash, 2))
                + " | shares " + str(round(self.account.shares, 4))
                + " | equity " + str(round(self.equity[-1], 2)))

    def event_line(self):
        """The latest event: the last order and the desk's reply to it."""
        return "last order: " + self.last_order + " -> " + (self.last_reply or "-")

    # ------------------------------------------------------------ moving through time

    def _advance_one_day(self):
        """Move the calendar one day on and record that day's closing equity."""
        self.day = self.day + 1
        self.equity.append(self.account.value(self.close_today()))

    def step(self):
        """Step one day with NO trading (the 'step one day' menu item)."""
        if self.at_end():
            raise SimulationError("already on the last day (" + str(self.last_day()) + ")")
        self._advance_one_day()

    # ------------------------------------------------------------------ manual trades

    def manual_trade(self, action, amount):
        """A trade typed at the desk. Fills at TODAY'S CLOSE. Returns the reply string."""
        price = self.close_today()
        before = self.account.shares
        try:
            reply = self.account.execute((action, amount), price)
        except ValueError as error:
            raise SimulationError(str(error))
        self._log_fill(self.day, action, price, before)
        self.last_order = action + " " + str(amount) + " (manual) at close " + str(price)
        self.last_reply = reply
        # The trade changed cash and shares (and cost a fee), so today's equity changes.
        # Replace today's entry instead of adding a new one: it is still the same day.
        self.equity[-1] = self.account.value(price)
        self.run_finished = False         # a hand-made trade means this is no longer a pure agent run
        return reply

    def _log_fill(self, day, action, price, shares_before):
        """Write a fill into self.trades, but only if shares really changed hands."""
        change = self.account.shares - shares_before
        if change > 0:
            self.trades.append((day, "BUY", price, change))
        elif change < 0:
            self.trades.append((day, "SELL", price, -change))

    # -------------------------------------------------------------------- the agent

    def agent_day(self, agent, use_prices=True, use_news=True):
        """Let the agent decide on today, fill its order tomorrow, move to tomorrow.

        use_prices False -> the agent gets bar=None   (the 'news-only' regime)
        use_news   False -> the agent gets headlines=None (the 'prices-only' regime)
        """
        if self.at_end():
            raise SimulationError("no day left: an order decided today would fill on a "
                                  "day that does not exist")
        today = self.market.bars[self.day]
        if use_prices:
            bar = dict(today)            # a COPY: the agent cannot edit our data
        else:
            bar = None
        if use_news:
            headlines = list(self.market.headlines_on(self.day))
        else:
            headlines = None

        # The agent is asked for an order. It only ever sees today and the past it stored.
        try:
            order = agent.decide(bar, headlines, self.account.cash, self.account.shares,
                                 self.last_response)
        except Exception as error:
            raise SimulationError("the agent crashed on day " + str(self.day) + ": "
                                  + type(error).__name__ + ": " + str(error))

        # The fill price is TOMORROW'S OPEN: the first price after the decision.
        fill_price = self.market.bars[self.day + 1]["open"]
        before = self.account.shares
        try:
            reply = self.account.execute(order, fill_price)
        except ValueError as error:
            raise SimulationError("the agent returned a bad order on day " + str(self.day)
                                  + ": " + str(error))
        self._log_fill(self.day + 1, order[0], fill_price, before)
        self.last_order = str(order[0]) + " " + str(order[1]) + " (agent) at open " + str(fill_price)
        self.last_reply = reply
        self.last_response = reply            # the agent sees this on its NEXT call
        self._advance_one_day()               # tomorrow arrives; equity uses tomorrow's CLOSE

    def run_agent(self, agent_class, use_prices=True, use_news=True, days=None):
        """Run a fresh agent from today to the last day. Returns nothing; read self.equity.

        The competition builds the agent as TradingAgent() with no arguments, and seeds
        both random generators the same way before every run; we copy both habits so the
        desk behaves like the competition. The loop does one agent call and one fill per
        day, so the whole back-test is O(n) in the number of days n.
        """
        random.seed(12345)
        np.random.seed(12345)
        try:
            agent = agent_class()
        except Exception as error:
            raise SimulationError("could not create the agent: " + type(error).__name__
                                  + ": " + str(error))
        self.last_response = None                 # first call of a run gets None
        self.run_finished = False
        self.run_saved = False
        self.agent_name = agent_class.__name__
        if use_prices and use_news:
            self.regime = "both"
        elif use_prices:
            self.regime = "prices only"
        else:
            self.regime = "news only"
        if days is not None:
            if isinstance(days, bool) or int(days) < 1:
                raise SimulationError("the number of agent days must be at least 1")
            days = min(int(days), self.last_day() - self.day)
        completed = 0
        while not self.at_end() and (days is None or completed < days):
            self.agent_day(agent, use_prices, use_news)
            completed += 1
        self.run_finished = self.at_end()
        if self.run_finished:
            self.run_saved = False
