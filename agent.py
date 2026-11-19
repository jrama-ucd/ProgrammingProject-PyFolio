# agent.py -- the competition entry. Self-contained: it imports NOTHING and reads no files.
#
# 1) Put your group's code in the GROUP_CODE line (it arrived by email after the group lock).
# 2) Replace the two comment lines under it with every member's full name and the member
#    who is this file's reference.
# Then run:  python3 agent_unittest.py   and   python3 selfcheck.py

GROUP_CODE = "G03-RGAD"    
# Group: Ayesha Tariq, Noaman Hakim, Muhammad Anas Asif, Justin Ramaputra, Mashhood Ahmad
# Reference: Muhammad Anas Asif

# WHAT THIS AGENT DOES, IN TWO SENTENCES
# ---------------------------------------
# It reads today's headlines, scores each word with the sentiment lexicon (good word = +1,
# bad word = -1), and adds the scores up into a "mood". Positive mood -> be in the market,
# negative mood -> be out of it, zero mood -> change nothing.
#
# WHY NEWS AND NOT PRICE
# ----------------------
# On the four training markets, today's headline mood correlates with tomorrow's return
# (about +0.13 to +0.23 on every market), while simple price rules (moving-average
# crossover, yesterday's move, distance from the 20-day mean) showed no consistent edge
# (correlations all under 0.08, and the sign changed from market to market). A
# moving-average crossover also lost money in a back-test. So with prices only the agent
# does the safe thing and holds.
#
# THE THREE REGIMES (agent contract section 3)
#   prices only   headlines is None -> HOLD (nothing to read, and no proven price edge)
#   news only     bar is None       -> trade on the mood with all-in / all-out orders,
#                                      because without a price we cannot count shares
#   both          price AND news    -> same mood signal, but the price lets us size the
#                                      position: we never put more than MAX_EXPOSURE of our
#                                      money into shares, which keeps drawdowns smaller
#
# THE RULES OF THE CONTRACT THAT THIS FILE KEEPS
#   * no imports at all (so nothing outside numpy/pandas/math/random can sneak in)
#   * no file reading or writing, no input(), no eval/exec
#   * no randomness: the same data always gives the same orders (deterministic)
#   * never crashes on bar=None or headlines=None
#   * returns a (action, amount) tuple every single day

# The lexicon: the 28 non-zero words of sentiment_words.csv, copied in because an agent
# is not allowed to read files. The 12 neutral words of that file score 0, so leaving them
# out changes nothing: a word that is not in this dict also counts as 0.
WORD_SCORES = {
    "surge": 1.0, "rally": 1.0, "soar": 1.0, "gain": 1.0, "bullish": 1.0,
    "breakout": 1.0, "upgrade": 1.0, "beat": 1.0, "record": 1.0, "boom": 1.0,
    "jump": 1.0, "optimism": 1.0, "inflow": 1.0, "adoption": 1.0,
    "crash": -1.0, "plunge": -1.0, "slump": -1.0, "loss": -1.0, "bearish": -1.0,
    "selloff": -1.0, "downgrade": -1.0, "miss": -1.0, "fear": -1.0, "dump": -1.0,
    "drop": -1.0, "panic": -1.0, "outflow": -1.0, "ban": -1.0,
}

# The most we put into shares when we can see the price, as a fraction of our cash.
# 0.75 means a quarter of the money always stays in cash as a cushion.
MAX_EXPOSURE = 0.75

# The 0.1% trading fee. A share really costs price * (1 + FEE) when we buy it.
FEE = 0.001

# Characters stripped from the edges of a word, so "surge," or "surge:" still match "surge".
PUNCTUATION = ".,:;!?\"'()"


class TradingAgent:
    """The strategy. The engine calls decide() once per day and the agent only REQUESTS
    a trade; the desk's Account is the one that accepts or rejects it."""

    def __init__(self):
        # The competition builds the agent with no arguments, so there are none.
        # The agent is stateless: today's decision depends only on today's inputs, so
        # there is no history list to keep (the contract says you only get today).
        self.word_scores = WORD_SCORES

    def mood_of(self, headlines):
        """Add up the word scores of every word in every headline of the day.

        Example: ["ZNTH surge as inflow boom", "ZNTH fear of outflow"] -> 3 - 2 = +1.
        The loop is one pass over all words of the day's headlines: a handful of words
        a day, so it is O(1) per day for this data.
        """
        mood = 0.0
        for line in headlines:
            # lower() so 'Surge' = 'surge'; split() cuts the line into words on spaces.
            for word in str(line).lower().split():
                word = word.strip(PUNCTUATION)
                mood = mood + self.word_scores.get(word, 0.0)   # unknown word -> 0
        return mood

    def decide(self, bar, headlines, cash, shares, last_response):
        """Called once per day. Returns ("BUY", n), ("SELL", n) or ("HOLD", 0).

        bar           today's prices dict, or None in the news-only regime
        headlines     today's headlines (a list), or None in the prices-only regime
        cash, shares  our balances: the truth, so we never track them ourselves
        last_response the desk's reply to our previous order. We do not need it: if an
                      order was rejected we are still not in the market, so the same
                      rule simply asks again tomorrow.
        """
        # Prices only: nothing to read -> do nothing.
        if headlines is None:
            return ("HOLD", 0)

        mood = self.mood_of(headlines)

        if mood > 0:                       # good news: we want to be in
            return self.go_in(bar, cash, shares)
        if mood < 0:                       # bad news: we want to be out
            return self.go_out(shares)
        return ("HOLD", 0)                 # no news or mixed news: leave things as they are

    def go_in(self, bar, cash, shares):
        """Buy, unless we are already in the market or cannot afford anything."""
        if shares > 0 or cash <= 0:
            return ("HOLD", 0)             # already holding shares (or nothing to spend)
        if bar is None:
            # News-only run: no price, so we cannot work out a share count.
            # (BUY, -1) means "as many as I can afford" and is never rejected for size.
            return ("BUY", -1)
        price = bar["close"]
        if not price > 0:                  # also catches NaN, which is never > 0
            return ("HOLD", 0)
        # How many whole shares does MAX_EXPOSURE of our cash buy? Each share costs
        # price * (1 + FEE). int() rounds DOWN, so we never ask for more than we can pay.
        # (We use today's close as the price; the order fills at tomorrow's open, and the
        # 25% cushion covers the difference.)
        count = int(MAX_EXPOSURE * cash / (price * (1.0 + FEE)))
        if count < 1:
            return ("HOLD", 0)
        return ("BUY", count)

    def go_out(self, shares):
        """Sell everything we hold, if we hold anything."""
        if shares > 0:
            return ("SELL", -1)            # -1 = sell all my shares
        return ("HOLD", 0)