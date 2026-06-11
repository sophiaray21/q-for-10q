import numpy as np
from actions import N_ACTIONS

'''
The Q-network for the agent, written from scratch in numpy (no pytorch/tensorflow allowed
for the final submission). Architecture: state vector (11 features) -> two hidden layers
with ReLU -> 3 linear outputs, one Q-value per action (sell, hold, buy).

The network doesn't know anything about trading, it's just a function approximator:
it takes a state and estimates how good each action is from that state.
'''


class DQN:
    def __init__(self, state_dim=11, hidden1=64, hidden2=32, n_actions=N_ACTIONS, lr=0.001, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization - scales the random weights by sqrt(2/fan_in) so the
        # signal doesn't blow up or die going through the ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2.0 / state_dim), (state_dim, hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2))
        self.b2 = np.zeros(hidden2)
        # output layer gets a much smaller init than He on purpose: quarterly rewards
        # live around +/-0.05, and if the initial Q-values come out at magnitude ~1 the
        # network wastes thousands of updates just shrinking them toward zero before any
        # buy/sell signal can be learned. Small init = Q starts near 0 = "knows nothing".
        self.W3 = rng.normal(0, 0.01, (hidden2, n_actions))
        self.b3 = np.zeros(n_actions)
        self.lr = lr

    def forward(self, states):
        '''
        states: (batch, state_dim) array. Returns (q_values, cache).
        cache holds the intermediate activations because backprop needs them.
        '''
        states = np.atleast_2d(states)
        z1 = states @ self.W1 + self.b1
        a1 = np.maximum(0, z1)            # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        q = a2 @ self.W3 + self.b3        # linear output, Q-values can be negative
        cache = (states, z1, a1, z2, a2)
        return q, cache

    def predict(self, states):
        q, _ = self.forward(states)
        return q

    def backward(self, cache, q_pred, actions, targets):
        '''
        One gradient step. Loss is MSE but ONLY on the Q-value of the action that was
        actually taken - we don't have a training signal for the other two actions.

        actions: (batch,) ints, targets: (batch,) floats (the TD targets).
        Returns the loss so the training loop can track it.
        '''
        states, z1, a1, z2, a2 = cache
        batch = states.shape[0]

        # dL/dq is zero everywhere except the taken action's entry
        dq = np.zeros_like(q_pred)
        idx = np.arange(batch)
        diff = q_pred[idx, actions] - targets
        dq[idx, actions] = 2.0 * diff / batch

        # chain rule back through the layers
        dW3 = a2.T @ dq
        db3 = dq.sum(axis=0)
        da2 = dq @ self.W3.T
        dz2 = da2 * (z2 > 0)              # ReLU gradient: 1 where input was positive
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = states.T @ dz1
        db1 = dz1.sum(axis=0)

        # plain SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        return float(np.mean(diff ** 2))

    def copy_from(self, other):
        '''Used to sync the target network with the online network.'''
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
        self.W3 = other.W3.copy(); self.b3 = other.b3.copy()


if __name__ == "__main__":
    # sanity check 1: shapes
    net = DQN(seed=42)
    q, _ = net.forward(np.random.default_rng(0).normal(size=(5, 11)))
    print("q shape (should be (5, 3)):", q.shape)

    # sanity check 2: numerical gradient check on one weight.
    # nudge W3[0,0] up and down, the change in loss should match the analytic gradient.
    rng = np.random.default_rng(1)
    s = rng.normal(size=(4, 11))
    a = np.array([0, 1, 2, 1])
    t = rng.normal(size=4)

    def loss_at(w):
        saved = net.W3[0, 0]
        net.W3[0, 0] = w
        qv, _ = net.forward(s)
        out = np.mean((qv[np.arange(4), a] - t) ** 2)
        net.W3[0, 0] = saved
        return out

    eps = 1e-5
    w0 = net.W3[0, 0]
    numeric = (loss_at(w0 + eps) - loss_at(w0 - eps)) / (2 * eps)

    qv, cache = net.forward(s)
    dq = np.zeros_like(qv)
    dq[np.arange(4), a] = 2.0 * (qv[np.arange(4), a] - t) / 4
    analytic = (cache[4].T @ dq)[0, 0]
    print(f"gradient check: numeric={numeric:.8f} analytic={analytic:.8f}")
    assert abs(numeric - analytic) < 1e-6, "backprop math is off"
    print("gradient check passed")
