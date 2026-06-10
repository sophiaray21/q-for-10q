import numpy as np
from trading_agent import TradingAgent
from reward import reward

'''
Main training loop (ticket #12). Runs the DQN agent over historical stock-quarters.

The real state vectors come from ticket #4 (financials + market data + sentiment merged
into ~11 features per stock-quarter). Since thats still in progress, this loop reads
whatever csv #4 produces later through load_states(), and until then make_fake_states()
generates placeholder data with a learnable pattern baked in so we can verify the whole
loop works end to end.

An episode = one stock's quarters in chronological order. Each quarter the agent picks
sell/hold/buy, the reward comes from the NEXT quarter's price (see reward.py), we store
the transition in the replay buffer and do one training step.
'''


def make_fake_states(n_stocks=10, n_quarters=40, seed=0):
    # placeholder until issue #4 is done: feature 0 secretly predicts next quarter's
    # direction, the other 10 features are noise. if the loop cant learn THIS, something
    # is broken, so it doubles as a test of the whole training stack.
    rng = np.random.default_rng(seed)
    stocks = {}
    for s in range(n_stocks):
        states = rng.normal(size=(n_quarters, 11))
        prices = [100.0]
        for t in range(n_quarters):
            drift = 0.04 if states[t, 0] > 0 else -0.04
            prices.append(prices[-1] * np.exp(drift + rng.normal(0, 0.02)))
        stocks["FAKE%d" % s] = (states, np.array(prices))
    return stocks


def load_states(csv_path):
    # for when #4 lands. expected columns: ticker, quarter_end, close, then the feature
    # columns (anything starting with f). rows should be one stock-quarter each.
    import pandas as pd
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    stocks = {}
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("quarter_end")
        stocks[ticker] = (g[feature_cols].to_numpy(dtype=float),
                          g["close"].to_numpy(dtype=float))
    return stocks


def train(stocks, epochs=30, seed=7, verbose=True):
    state_dim = next(iter(stocks.values()))[0].shape[1]
    agent = TradingAgent(state_dim=state_dim, seed=seed)
    history = []
    for epoch in range(epochs):
        total_reward = 0.0
        losses = []
        for ticker, (states, prices) in stocks.items():
            prev_pos = 0
            T = min(len(states), len(prices) - 1)
            for t in range(T):
                a = agent.act(states[t])
                r, prev_pos = reward(a, prices[t], prices[t + 1], prev_pos)
                done = (t == T - 1)
                s_next = states[t + 1] if not done else states[t]
                agent.remember(states[t], a, r, s_next, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                total_reward += r
        history.append(total_reward)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:>3}: total reward {total_reward:+8.3f}   eps {agent.eps:.3f}"
                  f"   avg loss {np.mean(losses):.5f}")
    return agent, history


def save_weights(agent, path="dqn_weights.npz"):
    n = agent.online
    np.savez(path, W1=n.W1, b1=n.b1, W2=n.W2, b2=n.b2, W3=n.W3, b3=n.b3)
    print("saved weights to", path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        stocks = load_states(sys.argv[1])
        print(f"loaded state vectors for {len(stocks)} tickers from {sys.argv[1]}")
    else:
        stocks = make_fake_states()
        print("no csv given - training on placeholder data (run with the #4 csv once it exists)")
    agent, hist = train(stocks)
    save_weights(agent)
    print(f"reward first epoch {hist[0]:+.3f} -> last epoch {hist[-1]:+.3f}"
          " (should clearly go up on the placeholder data)")
