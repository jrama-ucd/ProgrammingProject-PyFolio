# desk.py -- START THE PROGRAM WITH:   python3 desk.py     (Windows: python desk.py)
# Reference: Ayesha Tariq, Noaman Hakim, Muhammad Anas Asif, Justin Ramaputra, Mashhood Ahmad (Divided equally)
#
# WHAT THIS FILE IS
# -----------------
# The text menu ("the trading desk"). It is the same menu-loop pattern as the cafe
# register from the lectures:  while choice != "0":  show menu, read a choice, act.
#
# The desk itself holds NO money logic and NO file formats. It only:
#   1. reads what the user types (and refuses daft input politely),
#   2. calls the right object:  data_loader (load), Simulator (step, trade, run agent),
#      analytics (numbers, charts), Storage (save / load),
#   3. prints the result.
# The Account (inside the Simulator) owns the money and can refuse an order; the agent
# only requests trades.
#
# COMMAND LINE (every setting has a default, so "python3 desk.py" with no arguments works):
#   python3 desk.py [--ticker ZNTH] [--cash 10000] [--data data/training/]
#
# Allowed libraries here: sys (for the command line) plus our own modules.

import sys

import analytics
from account import is_number
from analytics import AnalyticsError, RunStats
from data_loader import DataError, DataNotFound, load_market
from simulator import SimulationError, Simulator
from storage import Storage, StorageError

# The agent is imported inside try/except so that a broken agent.py shows a message
# instead of stopping the whole desk. TradingAgent is None when it could not be loaded.
try:
    from agent import TradingAgent
    AGENT_PROBLEM = ""
except Exception as import_error:               # includes SyntaxError inside agent.py
    TradingAgent = None
    AGENT_PROBLEM = type(import_error).__name__ + ": " + str(import_error)

# Defaults for every setting (the brief: "give every command-line setting a default").
DEFAULT_CASH = 10000.0
DEFAULT_SHARES = 0.0

USAGE = """Usage: python3 desk.py [--ticker NAME] [--cash AMOUNT] [--data FOLDER]
  --ticker  market to load at start, e.g. ZNTH         (default: choose from the menu)
  --cash    starting wallet, at least 10000          (default: 10000)
  --shares  starting shares, 0 or more                 (default: 0)
  --data    folder holding NAME.csv and NAME_news.txt  (default: data/training/)"""


class QuitDesk(Exception):
    """Raised when the input stream ends (Ctrl-D / closed pipe) so the desk can say goodbye."""


# ----------------------------------------------------------------------------------
# Small input helpers. Every prompt in the program goes through these, so bad input is
# handled in one place.
# ----------------------------------------------------------------------------------

def read_line(prompt):
    """input() that turns end-of-input into QuitDesk instead of a traceback."""
    try:
        return input(prompt)
    except EOFError:
        raise QuitDesk()


def ask_number(prompt, default, minimum=None, allow_max=False):
    """Ask until the user types a valid number. Enter alone gives the default.

    minimum   if given, values below it are refused ("-500" as starting cash is refused)
    allow_max if True, -1 is accepted as a special value meaning "all / max"
    Python's float() understands things like 1e4 (= 10000.0), so that is accepted too.
    """
    while True:
        text = read_line(prompt).strip()
        if text == "":
            return float(default)
        try:
            value = float(text)
        except ValueError:
            print("  ! '" + text + "' is not a number. Try again (for example 100 or 2.5).")
            continue
        if not is_number(value):
            print("  ! Please type a real number (not nan or inf).")
            continue
        if allow_max and value == -1:
            return value
        if minimum is not None and value < minimum:
            if allow_max:
                print("  ! Type a number above 0, or -1 for all/max.")
            else:
                print("  ! That must be " + str(minimum) + " or more. Try again.")
            continue
        return value


def parse_arguments(argv):
    """Read the command line by hand (sys.argv). Returns a settings dict, or None for --help.

    Raises ValueError with a readable message for a bad flag or value.
    """
    settings = {"cash": DEFAULT_CASH, "shares": 0.0, "data": None, "ticker": None}
    i = 1                                       # argv[0] is the program name itself
    while i < len(argv):
        flag = argv[i]
        if flag in ("-h", "--help"):
            return None
        if flag not in ("--ticker", "--cash", "--shares", "--data"):
            raise ValueError("unknown option '" + flag + "'")
        if i + 1 >= len(argv):
            raise ValueError("option " + flag + " needs a value after it")
        value = argv[i + 1]
        if flag in ("--cash", "--shares"):
            try:
                number = float(value)
            except ValueError:
                raise ValueError(flag + " needs a number, got '" + value + "'")
            minimum = 0
            if flag == "--cash":
                minimum = 10000
            if not is_number(number) or number < minimum:
                raise ValueError(flag + " must be " + str(minimum) + " or more, got '" + value + "'")
            if flag == "--cash":
                settings["cash"] = number
            else:
                settings["shares"] = 0.0
        else:
            settings[flag[2:]] = value
        i = i + 2
    return settings


# ----------------------------------------------------------------------------------
# The desk
# ----------------------------------------------------------------------------------

class Desk:
    """The menu and its actions. self.sim is None until a dataset has been loaded."""

    def __init__(self, settings):
        self.cash = settings["cash"]            # starting wallet for the next load / reset
        self.shares = 0.0
        self.data_folder = settings["data"]     # None = look in the default folders
        self.storage = Storage()
        self.sim = None
        # menu number -> the method that does it (a dict instead of a long if/elif chain)
        self.actions = {
            "1": self.load_dataset,
            "2": self.set_balances,
            "3": self.step_days,
            "4": self.manual_buy,
            "5": self.manual_sell,
            "6": self.run_agent,
            "7": self.reports_and_analytics,
            "8": self.save_or_load,
        }

    # ------------------------------------------------------------- the main loop

    def run(self, first_ticker=None):
        """Show the menu until the user types 0."""
        self.user_name = read_line("  Enter your name: ").strip()
        if not self.user_name:
            self.user_name = "Trader"
        print()
        print("Welcome to Glacial Pace Capital, " + self.user_name + "!")
        print("Your starting wallet is " + format(self.cash, ".2f") + ".")
        try:
            if first_ticker:
                self.guarded(self.load_dataset, first_ticker)
                self.show_status()
            choice = ""
            while choice != "0":
                self.show_menu()
                choice = read_line("  Choose: ").strip()
                if choice == "0":
                    break
                action = self.actions.get(choice)
                if action is None:
                    print("  ! Please type a number from the menu.")
                    continue
                self.guarded(action)
                self.show_status()
        except (QuitDesk, KeyboardInterrupt):
            print()
        print("Goodbye.")

    def guarded(self, action, argument=None):
        """Run one menu action; any known problem becomes a one-line message, never a traceback."""
        try:
            if argument is None:
                action()
            else:
                action(argument)
        except (DataError, SimulationError, StorageError, AnalyticsError) as error:
            print("  ! " + str(error))
        except ValueError as error:
            print("  ! " + str(error))
        except (QuitDesk, KeyboardInterrupt):
            raise
        except Exception as error:
            # Last safety net: something we did not foresee. Say so plainly and carry on.
            print("  ! Unexpected problem (" + type(error).__name__ + ": " + str(error)
                  + "). Nothing was saved; you can carry on.")

    def show_menu(self):
        print()
        print("========== Glacial Pace Capital trading desk ==========")
        print("  [1] Load dataset            [2] Set starting amount to invest")
        print("  [3] Step trading days       [4] Manual buy")
        print("  [5] Manual sell             [6] Run agent")
        print("  [7] Reports and analytics   [8] Save / load")
        print("  [0] Quit")
        print("=======================================================")

    def show_status(self):
        """Balances, the day, and the latest event (the last order and the reply to it)."""
        if self.sim is None:
            print("  (no dataset loaded yet: choose 1)")
            return
        print("  " + self.sim.status_line())
        print("  " + self.sim.event_line())

    def need_market(self):
        """Stop an action politely if nothing is loaded yet."""
        if self.sim is None:
            raise SimulationError("Load a dataset first (option 1).")

    # ------------------------------------------------------------------ 1. load

    def load_dataset(self, ticker=None):
        """Load a market by ticker and start a fresh simulation on it."""
        if ticker is None:
            ticker = read_line("  Ticker (e.g. VOLT, ZNTH, NOVA, QBIT): ")
        try:
            market = load_market(ticker, self.data_folder)
        except DataNotFound as error:
            # Not in the usual places: offer to look somewhere else (handy for a new market).
            print("  ! " + str(error))
            folder = read_line("  Folder to look in (Enter to cancel): ").strip()
            if folder == "":
                return
            market = load_market(ticker, folder)
            self.data_folder = folder
        self.sim = Simulator(market, self.cash, self.shares)
        print("  Loaded " + market.ticker + " (" + str(len(market.bars)) + " days).")
        if not market.news_found:
            print("  Note: there is no news file for " + market.ticker
                  + ", so every day has no headlines.")

    # ------------------------------------------------------------ 2. balances

    def set_balances(self):
        """Set the starting wallet. Shares always start at zero."""
        cash = ask_number("  Starting amount to invest [" + str(self.cash) + "]: ",
                          self.cash, minimum=10000)
        self.cash = cash
        self.shares = 0.0
        if self.sim is not None:
            self.sim.reset(cash, 0.0)
            print("  Restarted from day 0 with wallet " + format(cash, ".2f")
                  + " and shares 0.")
        else:
            print("  Starting wallet set to " + format(cash, ".2f")
                  + ". Shares will start at 0 when you load a dataset.")

    # ----------------------------------------------------------------- 3. step

    def step_days(self):
        """Step forward without trading."""
        self.need_market()
        remaining = self.sim.last_day() - self.sim.day
        if remaining == 0:
            raise SimulationError("Already on the last day (" + str(self.sim.last_day()) + ").")
        count = ask_number("  How many days [1]: ", 1, minimum=1)
        count = min(int(count), remaining)       # never step past the end
        for _ in range(count):
            self.sim.step()

    # ------------------------------------------------------- 4 and 5. manual trades

    def manual_buy(self):
        self.manual_trade("BUY", "buy")

    def manual_sell(self):
        self.manual_trade("SELL", "sell")

    def manual_trade(self, action, word):
        """Manual trades fill at today's close; buys are entered as share quantities."""
        self.need_market()
        if action == "BUY":
            print("  Wallet: " + format(self.sim.account.cash, ".2f"))
            print("  Available to invest: " + format(self.sim.account.cash, ".2f"))
            while True:
                raw = read_line("  How many shares would you like to buy? ").strip()
                if raw == "":
                    print("  ! Please enter a share quantity. Leaving it blank will not buy anything.")
                    continue
                try:
                    amount = float(raw)
                except ValueError:
                    print("  ! Enter a valid number of shares.")
                    continue
                if not is_number(amount) or amount <= 0:
                    print("  ! Enter a number of shares greater than zero.")
                    continue
                break
            reply = self.sim.manual_trade("BUY", amount)
            print("  -> " + reply)
            if reply == "ACCEPT":
                print("  Wallet remaining: " + format(self.sim.account.cash, ".2f"))
        else:
            amount = ask_number("  How many shares to sell (-1 = all) [-1]: ", -1,
                                minimum=0.0000001, allow_max=True)
            reply = self.sim.manual_trade("SELL", amount)
            print("  -> " + reply)
            if reply == "ACCEPT":
                print("  Wallet after sale: " + format(self.sim.account.cash, ".2f"))

    # -------------------------------------------------------------- 6. run agent

    def run_agent(self):
        """Run the agent for a chosen number of days or until the dataset ends."""
        self.need_market()
        if TradingAgent is None:
            raise SimulationError("agent.py could not be loaded (" + AGENT_PROBLEM + ").")
        if self.sim.at_end():
            raise SimulationError("Already on the last day: restart with option 1 or 2.")

        print("  [1] Run for a specified number of days")
        print("  [2] Run until the end of the dataset")
        print("  [0] Cancel")
        duration_choice = read_line("  Choose: ").strip()
        if duration_choice == "0":
            print("  Agent run cancelled.")
            return
        if duration_choice == "1":
            remaining = self.sim.last_day() - self.sim.day
            days = int(ask_number("  Number of days to run: ", 1, minimum=1))
            days = min(days, remaining)
        elif duration_choice == "2":
            days = None
        else:
            raise SimulationError("Choose 1, 2 or 0.")

        print("  [1] Prices and news")
        print("  [2] Prices only")
        print("  [3] News only")
        mode = read_line("  What should the agent see? ").strip()
        if mode == "1":
            use_prices, use_news = True, True
        elif mode == "2":
            use_prices, use_news = True, False
        elif mode == "3":
            use_prices, use_news = False, True
        else:
            raise SimulationError("Choose 1, 2 or 3.")

        if days is None:
            actual_days = self.sim.last_day() - self.sim.day
        else:
            actual_days = days
        print("  Running agent for " + str(actual_days) + " day(s) ...")
        self.sim.run_agent(TradingAgent, use_prices, use_news, days=days)
        print("  Agent run result:")
        print(self.current_stats())
        if self.sim.at_end():
            print("  The agent has reached the end of the dataset.")
        else:
            print("  Agent stopped at day " + str(self.sim.day) + ".")

    def run_name(self):
        """'TradingAgent on ZNTH', plus the regime when the agent did not see everything."""
        name = self.sim.agent_name + " on " + self.sim.market.ticker
        if self.sim.regime not in ("", "both"):
            name = name + " (" + self.sim.regime + ")"
        return name

    def current_stats(self):
        """The three metrics, the final equity and the trade count of the run so far."""
        return RunStats(self.run_name(), self.sim.equity, len(self.sim.trades))

    # --------------------------------------------------------- 7. reports and analytics

    def reports_and_analytics(self):
        self.need_market()
        while True:
            print()
            print("========== Reports and analytics ==========")
            print("  [1] Plain-English management report")
            print("  [2] Stock price and trading activity chart")
            print("  [3] Portfolio value chart")
            print("  [4] Portfolio decline (drawdown) chart")
            print("  [5] Detailed performance figures")
            print("  [0] Back to main menu")
            choice = read_line("  Choose: ").strip()
            if choice == "0":
                return
            if choice == "1":
                self.management_report()
            elif choice == "2":
                analytics.plot_price_with_trades(
                    self.sim.market.ticker, self.sim.market.bars, self.sim.trades)
            elif choice == "3":
                analytics.plot_equity(self.sim.market.ticker, self.sim.equity)
            elif choice == "4":
                analytics.plot_drawdown(self.sim.market.ticker, self.sim.equity)
            elif choice == "5":
                print(self.current_stats())
            else:
                print("  ! Choose a number from 0 to 5.")

    def management_report(self):
        """Print a concise report in plain English for a non-technical reader."""
        stats = self.current_stats()
        wallet = self.sim.account.cash
        shares = self.sim.account.shares
        equity = self.sim.equity[-1]
        start_value = self.sim.equity[0]
        change = equity - start_value
        if change > 0:
            performance = "increased"
        elif change < 0:
            performance = "decreased"
        else:
            performance = "remained unchanged"
        report = [
            "========== MANAGEMENT REPORT ==========",
            "Company: Glacial Pace Capital",
            "Market: " + self.sim.market.ticker,
            "Period covered: day 0 to day " + str(self.sim.day),
            "Starting portfolio value: " + format(start_value, ".2f"),
            "Current wallet: " + format(wallet, ".2f"),
            "Shares currently held: " + format(shares, ".4f"),
            "Current portfolio value: " + format(equity, ".2f"),
            "Overall result: Portfolio value " + performance + " by "
                + format(abs(change), ".2f") + " ("
                + format(stats.total_return * 100, ".2f") + "%).",
            "Largest decline from a previous portfolio peak: "
                + format(abs(stats.max_drawdown) * 100, ".2f") + "%",
            "Completed trades: " + str(stats.trade_count),
            "======================================="
        ]
        report_text = "\\n".join(report)
        print(report_text)
        save = read_line("  Save this report to management_report.txt? [y/N]: ").strip().lower()
        if save in ("y", "yes"):
            report_file = open("management_report.txt", "w", encoding="utf-8")
            report_file.write(report_text + "\\n")
            report_file.close()
            print("  Report saved to management_report.txt.")

    # ------------------------------------------------------------------ 8. save / load

    def save_or_load(self):
        # Loading works even with nothing loaded; saving needs a market (checked in save()).
        choice = read_line("  [s]ave or [l]oad? ").strip().lower()
        if choice in ("s", "save"):
            self.save()
        elif choice in ("l", "load"):
            self.load()
        else:
            raise SimulationError("Type s to save or l to load.")

    def save(self):
        self.need_market()
        # A finished agent run is saved ONCE to results.txt; saving the same run twice
        # must not put it on the leaderboard twice.
        saved_run = False
        if self.sim.run_finished and not self.sim.run_saved:
            self.storage.save_run(self.current_stats())
            self.sim.run_saved = True
            saved_run = True
        self.storage.save_session(self.sim)
        print("  Saved the session to " + self.storage.session_file + ".")
        if saved_run:
            print("  Saved the run to " + self.storage.results_file + " and the leaderboard to "
                  + self.storage.leaderboard_file + ".")
        elif not self.sim.run_finished:
            print("  (No finished agent run to add to the leaderboard yet: use option 6.)")
        self.print_leaderboard()

    def load(self):
        data = self.storage.load_session()
        # The session remembers which folder the market came from; if it moved, search again.
        try:
            market = load_market(data["ticker"], data.get("folder") or None)
        except DataNotFound:
            market = load_market(data["ticker"], self.data_folder)
        sim = Simulator(market, data["start_cash"], data["start_shares"])
        sim.restore(data["day"], data["cash"], data["shares"], data["equity"], data["trades"],
                    data["start_cash"], data["start_shares"],
                    agent_name=data.get("agent_name", ""), regime=data.get("regime", ""),
                    run_finished=data.get("run_finished", False),
                    run_saved=data.get("run_saved", False),
                    last_order=data.get("last_order", "none yet"),
                    last_reply=data.get("last_reply", ""))
        self.sim = sim
        print("  Loaded session: " + market.ticker + " day " + str(sim.day) + ".")
        self.print_leaderboard()

    def print_leaderboard(self):
        print("  Leaderboard, best final equity first:")
        for line in self.storage.leaderboard_text():
            print("  " + line)


def main():
    """Read the command line, then run the desk."""
    try:
        settings = parse_arguments(sys.argv)
    except ValueError as error:
        print("  ! " + str(error))
        print(USAGE)
        return
    if settings is None:
        print(USAGE)
        return
    Desk(settings).run(settings["ticker"])


# Guarded main (week 5 layout: imports, constants, functions, then a guarded main): this
# line makes "python3 desk.py" start the program, while "import desk" (for tests) does not.
if __name__ == "__main__":
    main()
