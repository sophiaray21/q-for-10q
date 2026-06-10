import numpy as np

'''
Reward function for the trading agent (the math behind ticket #7).

Setup: the agent reads a stock's 10-Q quarter t and picks an action. The action sets
the position we hold from quarter t to t+1:

    sell -> -1 (short, we profit if the stock falls)
    hold ->  0 (flat / stay in cash)
    buy  -> +1 (long, we profit if the stock rises)

The reward arrives one quarter later:

    r_t = position * log(P_{t+1} / P_t) - cost * |position - prev_position|

Design choices and why:
- LOG return instead of percent return: log returns add up across quarters, so the sum
  of rewards over an episode equals the log of total portfolio growth. Maximizing total
  reward = maximizing compounded return, which is actually what we want.
- Transaction cost on position CHANGES only: holding a long position another quarter
  costs nothing, flipping from short to long costs twice the one-way cost (|-1 - 1| = 2).
  Without this the agent learns to churn positions every quarter, because there's no
  penalty stopping it.
- cost = 0.001 (10 bps one-way) is a standard rough assumption for liquid large caps.
'''

ACTION_TO_POSITION = {0: -1, 1: 0, 2: +1}   # sell, hold, buy
TRANSACTION_COST = 0.001


def reward(action, price_now, price_next, prev_position=0, cost=TRANSACTION_COST):
    '''
    action: 0/1/2 (sell/hold/buy), prices: this quarter's and next quarter's close,
    prev_position: what we were holding coming into this quarter (-1/0/+1).
    Returns (reward, new_position) - the caller keeps the position for the next step.
    '''
    position = ACTION_TO_POSITION[action]
    quarterly_log_return = np.log(price_next / price_now)
    trade_penalty = cost * abs(position - prev_position)
    return position * quarterly_log_return - trade_penalty, position


if __name__ == "__main__":
    # stock goes up 10%: buying should win, selling should lose, holding should be ~0
    for act, name in [(2, "buy"), (1, "hold"), (0, "sell")]:
        r, pos = reward(act, price_now=100, price_next=110)
        print(f"{name:5s} into a +10% quarter -> reward {r:+.4f} (position {pos:+d})")

    # churn check: flipping short->long should cost double the one-way fee
    r_flip, _ = reward(2, 100, 100, prev_position=-1)
    r_stay, _ = reward(2, 100, 100, prev_position=+1)
    print(f"flat quarter, flip short->long: {r_flip:+.4f} | already long: {r_stay:+.4f}")
