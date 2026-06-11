import numpy as np
from actions import BUY, random_action
from reward import reward
from train import make_fake_states, load_states, train

'''
Backtesting module (ticket #13). Walk-forward evaluation: train the agent ONLY on the
earlier part of each stock's history, then run it greedily (no exploration, no training)
on the later part it has never seen. Comparing against buy-and-hold and a random agent
is what tells us whether the DQN learned anything real or just got lucky.

Important detail: the split is by TIME, not random. Shuffling quarters across the split
would let the agent train on information from the future (look-ahead bias), which makes
a backtest meaningless. Thats also why load_states sorts by quarter_end.
'''


def split_stocks(stocks, train_frac=0.7):
    train_set, test_set = {}, {}
    for ticker, (states, prices) in stocks.items():
        cut = int(len(states) * train_frac)
        train_set[ticker] = (states[:cut], prices[:cut + 1])
        test_set[ticker] = (states[cut:], prices[cut:])
    return train_set, test_set


def run_policy(stocks, mode, agent=None, seed=0):
    # plays a policy over the test set, returns the per-quarter log returns.
    # modes: "dqn" (greedy agent), "buyhold" (always long), "random"
    rng = np.random.default_rng(seed)
    rets = []
    for ticker, (states, prices) in stocks.items():
        prev_pos = 0
        T = min(len(states), len(prices) - 1)
        for t in range(T):
            if mode == "dqn":
                a = agent.act(states[t], greedy=True)
            elif mode == "buyhold":
                a = BUY
            else:
                a = random_action(rng)
            r, prev_pos = reward(a, prices[t], prices[t + 1], prev_pos)
            rets.append(r)
    return np.array(rets)


def report(name, rets):
    total = np.exp(rets.sum()) - 1            # log returns add, so exp(sum) = total growth
    curve = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(curve)
    max_dd = ((curve - peak) / peak).min()    # worst drop from a previous high
    win = (rets > 0).mean()
    print(f"  {name:<13} total {total:+8.1%}   avg/qtr {rets.mean():+.4f}"
          f"   max drawdown {max_dd:+6.1%}   win rate {win:.0%}")


if __name__ == "__main__":
    import sys
    stocks = load_states(sys.argv[1]) if len(sys.argv) > 1 else make_fake_states()
    train_set, test_set = split_stocks(stocks)
    n_train = sum(len(s) for s, _ in train_set.values())
    n_test = sum(len(s) for s, _ in test_set.values())
    print(f"walk-forward split: {n_train} stock-quarters to train, {n_test} held out")

    agent, _ = train(train_set, epochs=30, verbose=False)

    print("\nout-of-sample results (the held-out 30%, never seen in training):")
    report("dqn (greedy)", run_policy(test_set, "dqn", agent=agent))
    report("buy and hold", run_policy(test_set, "buyhold"))
    report("random", run_policy(test_set, "random"))
    print("\nif dqn doesnt beat random, the agent learned nothing - thats the honest check")
