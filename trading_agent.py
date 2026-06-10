import numpy as np
from dqn import DQN
from replay_buffer import ReplayBuffer

'''
The decision-making layer (ticket #11). Glues together the Q-network, the target
network and the replay buffer into an agent you can ask for actions and train.

Two DQN tricks in here, both from Mnih et al. 2015:
- epsilon-greedy: with probability epsilon act randomly (explore), otherwise take the
  action with the highest Q-value (exploit). Epsilon decays over training so the agent
  explores early and settles down later.
- target network: the TD target uses a frozen copy of the network that only syncs every
  N steps. If the same network produced both the prediction and the target, the target
  would move every update and training chases its own tail.
'''

ACTIONS = {0: "sell", 1: "hold", 2: "buy"}


class TradingAgent:
    def __init__(self, state_dim=11, gamma=0.95, lr=0.005,
                 eps_start=1.0, eps_min=0.05, eps_decay=0.995,
                 buffer_capacity=10000, target_sync_every=100, seed=None):
        self.online = DQN(state_dim=state_dim, lr=lr, seed=seed)
        self.target = DQN(state_dim=state_dim, lr=lr, seed=seed)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.gamma = gamma                  # discount on future rewards
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_sync_every = target_sync_every
        self.train_steps = 0
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        '''epsilon-greedy. greedy=True for evaluation/backtest (no exploration).'''
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(0, 3))
        q = self.online.predict(state)
        return int(np.argmax(q[0]))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size=32):
        '''One gradient update from a random minibatch. Returns the loss (or None
        if the buffer doesn't have enough samples yet).'''
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # TD target: r + gamma * max_a' Q_target(s', a'), but just r on terminal steps
        q_next = self.target.predict(next_states)
        targets = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        q_pred, cache = self.online.forward(states)
        loss = self.online.backward(cache, q_pred, actions, targets)

        # housekeeping: decay exploration, periodically sync the target net
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self.target.copy_from(self.online)
        return loss


if __name__ == "__main__":
    # end-to-end smoke test on a fake pattern the agent should be able to learn:
    # if feature 0 is positive the stock goes up next quarter (buy is right),
    # if negative it goes down (sell is right). No real data needed to prove learning.
    rng = np.random.default_rng(7)
    agent = TradingAgent(seed=7)

    correct_before = 0
    test_states = rng.normal(size=(200, 11))
    for s in test_states:
        right = 2 if s[0] > 0 else 0
        if agent.act(s, greedy=True) == right:
            correct_before += 1

    for episode in range(5000):
        s = rng.normal(size=11)
        a = agent.act(s)
        up = s[0] > 0
        pos = {0: -1, 1: 0, 2: 1}[a]
        r = pos * (0.05 if up else -0.05)   # simplified reward, same shape as reward.py
        s_next = rng.normal(size=11)
        # done=True: each fake decision is its own one-step episode. With done=False the
        # TD target would bootstrap off a RANDOM unrelated next state, which is pure
        # noise - found that out the hard way (accuracy stalled at ~38%).
        agent.remember(s, a, r, s_next, True)
        agent.train_step()

    correct_after = 0
    for s in test_states:
        right = 2 if s[0] > 0 else 0
        if agent.act(s, greedy=True) == right:
            correct_after += 1

    print(f"greedy accuracy on the planted pattern: before training {correct_before/200:.0%}, "
          f"after 5000 steps {correct_after/200:.0%}")
    print(f"epsilon decayed to {agent.eps:.3f}, target net synced {agent.train_steps // 100} times")
    assert correct_after > 150, "agent failed to learn the planted pattern"
    print("learning check passed")
