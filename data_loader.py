# data_loader.py
# Reference: <full name of the member who is the reference for the data piece>
#
# WHAT THIS FILE IS
# -----------------
# Everything about READING the market files lives here, and nothing else does
# (rubric item 5: "loading code isolated from strategy code").
#
# The rules from the dataset guide that this file follows:
#   * The files are read exactly as distributed, at run time, every time. We never
#     convert, clean, edit or re-save them, and we never need a file somebody prepared
#     earlier. So the held-back market (same schema) loads with no change.
#   * <TICKER>.csv       header day,open,high,low,close,volume , one row per day
#   * <TICKER>_news.txt  header day,headline , zero, one or many headlines per day
#   * A missing news file, or a day with no headline, must NOT crash the program.
#
# Libraries: pandas reads the CSV (taught: read_csv), numpy checks the numbers
# (taught: arrays). The news file is plain text, so it uses plain file reading
# (week 2/3), as the dataset guide says.

import numpy as np
import pandas as pd

# Folders we look in, in this order, when the user does not name one.
# selfcheck.py uses the same two places, so everything lines up.
DATA_FOLDERS = ("data/training/", "../data/training/")

# The columns the price file must have (the dataset guide's schema).
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class DataError(Exception):
    """Raised for any data problem. The message is written for the user to read."""


class DataNotFound(DataError):
    """The ticker's files are not in any folder we looked in (the desk can ask for a folder)."""


class Market:
    """One loaded market: the daily price bars and the news, kept together.

    bars  : list of dicts, one per day, each {"open","high","low","close","volume"}.
            The list index IS the day number (0, 1, 2, ...).
    news  : dict mapping day number -> list of headline strings.
    """

    def __init__(self, ticker, folder, bars, news, news_found=True):
        self.ticker = ticker
        self.folder = folder
        self.bars = bars
        self.news = news
        self.news_found = news_found     # False when there was no news file at all

    def headlines_on(self, day):
        """The headlines of one day. A day with no news gives an empty list, not an error."""
        return self.news.get(day, [])


def can_open(path):
    """True if the file exists and can be opened for reading."""
    try:
        handle = open(path, encoding="utf-8")
        handle.close()
        return True
    except (OSError, UnicodeDecodeError):
        return False


def find_folder(ticker, folder=None):
    """Return the folder that holds <ticker>.csv, or "" if it is nowhere we look.

    If the caller names a folder we look only there; otherwise we try DATA_FOLDERS.
    """
    if folder:
        if not folder.endswith("/") and not folder.endswith("\\"):
            folder = folder + "/"
        candidates = (folder,)
    else:
        candidates = DATA_FOLDERS
    for candidate in candidates:
        if can_open(candidate + ticker + ".csv"):
            return candidate
    return ""


def load_bars(path):
    """Read the price CSV into a list of bar dicts, checking it on the way.

    Raises DataError with a plain message for every problem a user could cause
    (wrong file, missing column, empty file, text in a number column, zero prices).
    """
    try:
        frame = pd.read_csv(path)                     # header row, comma separated
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataError("The price file " + path + " is empty or not a valid CSV ("
                        + str(error) + ").")
    except (OSError, UnicodeDecodeError) as error:
        raise DataError("Could not read " + path + ": " + str(error))

    # The schema check: every column we need must be there.
    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            raise DataError("The price file " + path + " has no '" + column
                            + "' column. Expected: day,open,high,low,close,volume.")
    if len(frame) < 2:
        raise DataError("The price file " + path + " needs at least 2 days of data.")

    # Turn each column into a numpy ARRAY first (the contract's "convert before you slice"
    # warning): arrays work by position, a raw pandas Series works by label.
    columns = {}
    for column in PRICE_COLUMNS:
        try:
            values = np.array(frame[column], dtype=float)
        except (TypeError, ValueError):
            raise DataError("The '" + column + "' column in " + path
                            + " has something that is not a number.")
        if np.isnan(values).any():
            row = int(np.argmax(np.isnan(values)))
            raise DataError("The '" + column + "' column in " + path
                            + " has a missing value on row " + str(row) + ".")
        columns[column] = values
    if (columns["open"] <= 0).any() or (columns["close"] <= 0).any():
        raise DataError("The price file " + path + " has an open or close price that is 0 "
                        "or negative.")

    # Build one dict per day. The loop is one pass over the rows: O(n).
    bars = []
    for i in range(len(frame)):
        bar = {}
        for column in PRICE_COLUMNS:
            bar[column] = float(columns[column][i])
        bars.append(bar)
    return bars


def load_news(path):
    """Read the news file into {day: [headline, ...]}.

    Returns (news, found). found is False if the file does not exist; we then carry on
    with no news at all (the 'no news' regime) instead of crashing.
    A line that is not 'day,headline' is skipped, so one bad line cannot kill a run.
    """
    news = {}
    try:
        handle = open(path, encoding="utf-8")
    except OSError:
        return news, False
    is_header = True
    for line in handle:
        if is_header:                 # the first line is the header 'day,headline'
            is_header = False
            continue
        line = line.strip()
        if line == "":
            continue
        # split(",", 1) cuts at the FIRST comma only, so a comma inside a headline
        # survives (the dataset guide asks for exactly this pattern).
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            day = int(parts[0])
        except ValueError:
            continue
        if day not in news:
            news[day] = []
        news[day].append(parts[1].strip())
    handle.close()
    return news, True


def load_market(ticker, folder=None):
    """Load one market by ticker name, e.g. load_market("ZNTH"). Returns a Market.

    The only function the rest of the program needs from this file.
    """
    ticker = ticker.strip()
    if ticker == "":
        raise DataError("Type a ticker name such as ZNTH.")
    # Refuse anything that looks like a path so a ticker can only name a file in a
    # data folder (keeps the loader from wandering around the disk).
    if "/" in ticker or "\\" in ticker or ".." in ticker:
        raise DataError("A ticker is just a name such as ZNTH, not a path.")

    found = find_folder(ticker, folder)
    if found == "" and ticker != ticker.upper():
        ticker = ticker.upper()                       # tolerate 'znth' typed in lower case
        found = find_folder(ticker, folder)
    if found == "":
        if folder:
            places = folder
        else:
            places = " or ".join(DATA_FOLDERS)
        raise DataNotFound("Cannot find " + ticker + ".csv in " + places + ". Put the data "
                           "files in data/training/, or name the folder that holds them.")

    bars = load_bars(found + ticker + ".csv")
    news, news_found = load_news(found + ticker + "_news.txt")
    return Market(ticker, found, bars, news, news_found)
