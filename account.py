# account.py
# Reference: Muhammad Anas Asif
#
# WHAT THIS FILE IS
# -----------------
# The Account is the "authority" in the agent <-> interface split (project brief section 3).
# It is the ONLY place in the program where cash and shares change. The agent can only
# ASK for a trade; the Account decides whether the trade is allowed, charges the 0.1% fee,
# and answers with the exact reply text the agent contract demands:
#
#     "ACCEPT"
#     "REJECT currency balance: <your cash, to the cent>"
#     "REJECT stock owned: <how many shares you really hold>"
#
# The Account knows nothing about files, menus, charts or strategies. That is on purpose:
# one class, one job (week 7/8 refactor, rubric item 2).

# The trading fee: 0.1% of the trade value, charged on every buy and every sell.
FEE = 0.001

# The two reject prefixes are part of the spec, so they live in ONE place as constants.
REJECT_CASH = "REJECT currency balance: "
REJECT_STOCK = "REJECT stock owned: "

# The three actions an order may carry.
ACTIONS = ("BUY", "SELL", "HOLD")


def is_number(value):
    """True if value is a real, finite number (not text, not NaN, not infinity).

    NaN is the only value that is not equal to itself, which is why 'value == value'
    is a neat NaN test. 'abs(value) != float("inf")' rules out infinity.
    bool is excluded because True/False are technically ints in Python.
    """
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


class Account:
    """Holds the money (cash in USD) and the shares, and validates every order."""

    def __init__(self, cash=10000.0, shares=0.0):
        # assert/raise with a clear message: bad starting balances are refused here,
        # so the rest of the program can trust that cash and shares are sane numbers.
        if not is_number(cash) or cash < 0:
            raise ValueError("starting cash must be a number that is 0 or more, got "
                             + str(cash))
        if not is_number(shares) or shares < 0:
            raise ValueError("starting shares must be a number that is 0 or more, got "
                             + str(shares))
        self.cash = float(cash)
        self.shares = float(shares)

    def value(self, price):
        """Equity = cash + shares * price (the agent contract's definition, section 5)."""
        return self.cash + self.shares * price

    def execute(self, order, price):
        """Try to carry out one order at the given price. Returns the reply string.

        order is a tuple (action, amount), exactly what TradingAgent.decide returns:
            ("BUY", n)   buy n shares           ("SELL", n)  sell n shares
            ("BUY", -1)  buy as many as possible ("SELL", -1) sell everything
            ("HOLD", 0)  do nothing
        price is the price the order fills at. The CALLER chooses it:
            - the back-test passes tomorrow's OPEN   (no look-ahead)
            - a manual trade passes today's CLOSE    (the price is on the board)

        Rules (agent contract sections 4-6):
          * An order fills whole or not at all. A rejected order changes NOTHING.
          * Any negative amount counts as -1 ("max").
          * A -1 order is never rejected for size.
        """
        # --- 1. Check the order is well formed. A malformed order is a programming
        #        error in the caller, so we raise; the Simulator turns it into a
        #        friendly message instead of a traceback.
        if not isinstance(order, (tuple, list)) or len(order) != 2:
            raise ValueError("an order must be (action, amount), got " + str(order))
        action = order[0]
        if action not in ACTIONS:
            raise ValueError("unknown action " + str(action) + ", use BUY, SELL or HOLD")
        if not is_number(order[1]):
            raise ValueError("the amount must be a finite number, got " + str(order[1]))
        if not is_number(price) or price <= 0:
            raise ValueError("the price must be a number above 0, got " + str(price))
        # float() turns numpy numbers (np.int64 ...) into plain Python floats.
        amount = float(order[1])

        if action == "BUY":
            return self._buy(amount, price)
        if action == "SELL":
            return self._sell(amount, price)
        return "ACCEPT"          # HOLD: nothing to do

    def _buy(self, amount, price):
        """BUY logic, kept in its own function so execute() stays short."""
        cost_per_share = price * (1.0 + FEE)       # each share really costs price + 0.1%
        if amount < 0:
            # "buy as much as you can afford": the largest WHOLE number of shares.
            # int() chops the decimals off (rounds DOWN), never up, so we never overspend.
            # Worked example from the contract: cash 500, price 25 ->
            # 500 / 25.025 = 19.98 -> int -> 19 shares (not 20, which would cost 500.50).
            amount = int(self.cash / cost_per_share)
            # Guard for a floating point quirk: if the division rounded UP by one share
            # we hand that share back, so a -1 order can never be refused.
            if amount * cost_per_share > self.cash:
                amount = amount - 1
        if amount <= 0:
            return "ACCEPT"                         # nothing to buy: not an error
        cost = amount * cost_per_share
        if cost > self.cash:
            # Too expensive: refuse, change nothing, report the cash to the cent.
            return REJECT_CASH + str(round(self.cash, 2))
        self.cash = self.cash - cost
        self.shares = self.shares + amount
        return "ACCEPT"

    def _sell(self, amount, price):
        """SELL logic. Selling pays price minus the 0.1% fee."""
        if amount < 0:
            amount = self.shares                    # -1 means "sell everything I hold"
        if amount <= 0:
            return "ACCEPT"                         # nothing to sell: not an error
        if amount > self.shares:
            # You cannot sell what you do not own: refuse, report the real holding.
            return REJECT_STOCK + str(round(self.shares, 2))
        self.cash = self.cash + amount * price * (1.0 - FEE)
        self.shares = self.shares - amount
        return "ACCEPT"

    def __str__(self):
        """Readable text for printing an Account (week 7: __str__)."""
        return "cash " + str(round(self.cash, 2)) + " | shares " + str(round(self.shares, 4))
