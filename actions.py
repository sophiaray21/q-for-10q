'''
The action space, formalized (ticket #6). Single source of truth for what the agent
can do - every other module should import from here instead of redefining the
0/1/2 convention (reward.py, trading_agent.py, dqn.py and backtest.py previously
each carried their own copy, which only worked because they happened to agree).

Semantics:
- An action is chosen once per stock-quarter, when a new 10-Q becomes available
  (i.e. at the filing date - see build_states.py on why not the quarter end).
- The action sets the TARGET position held until the next decision point:
      sell -> -1  short one unit (profit if the stock falls)
      hold ->  0  flat / in cash
      buy  -> +1  long one unit (profit if the stock rises)
  This is full reallocation to the target, not an incremental trade on top of the
  current holding. Changing position costs transaction fees proportional to
  |new - old| (see reward.py), so flipping short->long pays double the one-way cost.
- The DQN's output layer has exactly one Q-value per action, indexed by these ids,
  so the integers double as network output indices. Reordering or extending them
  invalidates any saved weights (dqn_weights.npz).
'''

SELL, HOLD, BUY = 0, 1, 2
N_ACTIONS = 3

ACTION_NAMES = {SELL: "sell", HOLD: "hold", BUY: "buy"}
ACTION_TO_POSITION = {SELL: -1, HOLD: 0, BUY: +1}


def position_of(action):
    '''Target position (-1/0/+1) for an action id. Raises KeyError on garbage input
    rather than guessing - a bad action id upstream is a bug worth crashing on.'''
    return ACTION_TO_POSITION[action]


def random_action(rng):
    '''Uniform random action id, for epsilon-greedy exploration and random baselines.'''
    return int(rng.integers(0, N_ACTIONS))
