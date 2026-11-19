# storage.py
# Reference: Justin Ramaputra, Noaman Hakim
#
# WHAT THIS FILE IS
# -----------------
# basically saving and loading, in PLAIN TEXT, so results survive a restart (the same discipline as
# the cafe's revenue file from the lectures). Three files:
#
#   session.txt      the whole desk state, so you can quit and carry on later
#   results.txt      one line per saved run:   name,final_equity,trades,return,max_dd,sharpe
#   leaderboard.txt  rank,name,final_equity   one line per saved run, BEST FIRST
#
# The Storage class is the only code that knows how a line in those files looks. The menu
# just says "save this" / "load that". If the format ever changes, one file changes and the
# menu is untouched (single responsibility).
#
# This file uses open()/read/write, as taught in week 2.


class StorageError(Exception):
    """Raised when a file cannot be read or written."""


def clean_name(name):
    # A run's name goes into a comma-separated line, so commas and line breaks must go.
    return name.replace(",", " ").replace("\n", " ").replace("\r", " ").strip()


class Storage:
    # Reads and writes the three text files. File names are arguments with defaults.

    def __init__(self, session_file="session.txt", results_file="results.txt",
                 leaderboard_file="leaderboard.txt"):
        self.session_file = session_file
        self.results_file = results_file
        self.leaderboard_file = leaderboard_file

    # the session

    def save_session(self, simulator):
        """Write the state of a Simulator as 'key=value' lines.

        Floats are written with str(), which in Python 3 gives the shortest text that
        reads back as exactly the same number, so nothing drifts between save and load.
        """
        market = simulator.market
        lines = [
            "ticker=" + market.ticker,
            "folder=" + market.folder,
            "day=" + str(simulator.day),
            "start_cash=" + str(simulator.start_cash),
            "start_shares=" + str(simulator.start_shares),
            "cash=" + str(simulator.account.cash),
            "shares=" + str(simulator.account.shares),
            "agent_name=" + clean_name(simulator.agent_name),
            "regime=" + simulator.regime,
            "run_finished=" + str(int(simulator.run_finished)),
            "run_saved=" + str(int(simulator.run_saved)),
            "last_order=" + clean_name(simulator.last_order),
            "last_reply=" + clean_name(simulator.last_reply),
            "equity=" + ",".join([str(value) for value in simulator.equity]),
        ]
        for trade in simulator.trades:                  # trade=day,SIDE,price,shares
            lines.append("trade=" + str(trade[0]) + "," + trade[1] + ","
                         + str(trade[2]) + "," + str(trade[3]))
        self._write(self.session_file, lines)

    def load_session(self):
        """Read session.txt back into a plain dict. The desk rebuilds the objects from it.

        Raises StorageError with a clear message if the file is missing or damaged.
        """
        lines = self._read(self.session_file, "No saved session found (" + self.session_file
                           + "). Save one first with option 9.")
        data = {"trades": []}
        for number, line in enumerate(lines, start=1):
            line = line.strip()
            if line == "":
                continue
            parts = line.split("=", 1)
            if len(parts) != 2:
                raise StorageError(self.session_file + " line " + str(number)
                                   + " is not in the expected key=value format.")
            key, value = parts
            try:
                if key == "trade":
                    pieces = value.split(",")
                    data["trades"].append((int(pieces[0]), pieces[1],
                                           float(pieces[2]), float(pieces[3])))
                elif key == "equity":
                    data["equity"] = [float(item) for item in value.split(",")]
                elif key in ("day",):
                    data[key] = int(value)
                elif key in ("start_cash", "start_shares", "cash", "shares"):
                    data[key] = float(value)
                elif key in ("run_finished", "run_saved"):
                    data[key] = (value == "1")
                else:
                    data[key] = value                  # ticker, folder, agent_name, regime
            except (ValueError, IndexError):
                raise StorageError(self.session_file + " line " + str(number)
                                   + " has a value that cannot be read.")
        # A usable session must carry at least these keys.
        for key in ("ticker", "day", "cash", "shares", "equity", "start_cash", "start_shares"):
            if key not in data:
                raise StorageError(self.session_file + " is missing '" + key + "'.")
        return data

    # ------------------------------------------------------------------ the results

    def save_run(self, stats):
        """Add one line for a finished run to results.txt (append: nothing old is lost)."""
        line = (clean_name(stats.name) + "," + str(round(stats.final_equity, 2)) + ","
                + str(stats.trade_count) + "," + str(round(stats.total_return, 4)) + ","
                + str(round(stats.max_drawdown, 4)) + "," + str(round(stats.sharpe, 4)))
        try:
            handle = open(self.results_file, "a", encoding="utf-8")
            handle.write(line + "\n")
            handle.close()
        except OSError as error:
            raise StorageError("Could not write " + self.results_file + ": " + str(error))

    def read_results(self):
        """All saved runs as a list of (name, final_equity). No file yet -> empty list.

        A damaged line is skipped, not fatal: one bad line should not hide the others.
        """
        try:
            lines = self._read(self.results_file, "")
        except StorageError:
            return []
        results = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                results.append((parts[0], float(parts[1])))
            except ValueError:
                continue
        return results

    # ---------------------------------------------------------------- the leaderboard

    def write_leaderboard(self):
        """Rebuild leaderboard.txt from results.txt: best final equity first.

        The recipe is project document 09's: make [score, name] pairs, sort() (it orders
        by the first element), reverse() for best-first, then write one line each.

        Cost: reading the results is O(n), sorting is O(n log n), writing is O(n).
        So adding ONE result costs O(n log n) in the number of saved runs n, dominated
        by the sort. (Appending to results.txt alone is O(1).)
        """
        results = self.read_results()
        rows = [[final, name] for name, final in results]
        rows.sort()
        rows.reverse()
        lines = ["rank,name,final_equity"]
        for rank, row in enumerate(rows, start=1):
            lines.append(str(rank) + "," + row[1] + "," + str(round(row[0], 2)))
        self._write(self.leaderboard_file, lines)
        return rows

    def leaderboard_text(self):
        """The leaderboard as lines ready to print: '  1. TradingAgent on ZNTH 9224.48'."""
        rows = self.write_leaderboard()
        if len(rows) == 0:
            return ["  (no saved runs yet)"]
        return ["  " + str(rank) + ". " + row[1] + " " + str(round(row[0], 2))
                for rank, row in enumerate(rows, start=1)]

    # --------------------------------------------------------------- file plumbing

    def _write(self, filename, lines):
        """Write a list of lines, replacing the file. OSError becomes a friendly message."""
        try:
            handle = open(filename, "w", encoding="utf-8")
            handle.write("\n".join(lines) + "\n")
            handle.close()
        except OSError as error:
            raise StorageError("Could not write " + filename + ": " + str(error))

    def _read(self, filename, missing_message):
        """Read all lines of a file; a missing file raises StorageError(missing_message)."""
        try:
            handle = open(filename, encoding="utf-8")
            lines = handle.readlines()
            handle.close()
            return lines
        except FileNotFoundError:
            raise StorageError(missing_message or (filename + " does not exist."))
        except (OSError, UnicodeDecodeError) as error:
            raise StorageError("Could not read " + filename + ": " + str(error))
