import numpy as np
from collections import deque

'''
Experience replay buffer. The agent stores every (state, action, reward, next_state, done)
transition it sees, and trains on random minibatches from the buffer instead of only the
most recent step.

Why this matters (from the Mnih et al. 2015 DQN paper): consecutive quarters are
correlated, and training a network on correlated samples in order makes it overfit to
whatever regime it saw last. Sampling randomly breaks the correlation, and each
transition gets reused across many updates, which matters for us since quarterly data
is small to begin with.
'''


class ReplayBuffer:
    def __init__(self, capacity=10000, seed=None):
        self.buffer = deque(maxlen=capacity)   # old transitions fall off automatically
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.asarray(state, dtype=float), int(action),
                            float(reward), np.asarray(next_state, dtype=float), bool(done)))

    def sample(self, batch_size):
        '''Returns 5 stacked arrays ready to feed the network.'''
        picks = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in picks))
        return (np.stack(states), np.array(actions), np.array(rewards),
                np.stack(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=100, seed=0)
    for i in range(150):  # overfill past capacity on purpose
        buf.push(np.ones(11) * i, i % 3, float(i), np.ones(11) * (i + 1), False)
    print("len after 150 pushes into capacity-100 buffer (should be 100):", len(buf))
    s, a, r, ns, d = buf.sample(8)
    print("sampled shapes:", s.shape, a.shape, r.shape, ns.shape, d.shape)
    print("oldest transitions dropped (min reward should be >= 50):", r.min() >= 50)
