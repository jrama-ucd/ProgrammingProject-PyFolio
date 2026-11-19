MIS41110 trading desk - how to run it
=====================================

FOLDER LAYOUT (run everything from inside this folder)
  agent.py  desk.py  account.py  simulator.py  data_loader.py  analytics.py  storage.py  tests.py
  agent_unittest.py  selfcheck.py      <- the two course tools, do not edit
  data/training/                       <- the nine training files

BEFORE YOU DO ANYTHING: fill in the placeholders
  agent.py: the GROUP_CODE line, the "# Group:" line (every member's full name) and the
            "# Reference:" line.
  Every other .py file: the "# Reference:" line at the top.

COMMANDS (on Windows type  python  instead of  python3)
  python3 -m pip install numpy pandas matplotlib    (once, if they are not installed)
  python3 agent_unittest.py     protocol check of agent.py
  python3 selfcheck.py          the three "ran end-to-end" lines for the report
  python3 tests.py              this project's own asserts (creates test_*.txt, delete them after)
  python3 desk.py               the trading desk (text menu)
  python3 desk.py --ticker ZNTH --cash 5000 --shares 0 --data data/training/    (all optional)

FILES THE PROGRAM WRITES (your own outputs, safe to delete)
  session.txt  results.txt  leaderboard.txt  price_with_trades.png  equity_curve.png  drawdown.png

TO SUBMIT: agent.py plus every other .py file of the program, uploaded individually
(no zip, no folder), plus G??_report.pdf. NOT the data, agent_unittest.py or selfcheck.py.
