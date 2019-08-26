# Example 1
class GameState(object):
    def __init__(self):
        self.level = 0
        self.lives = 4


# Example 2
state = GameState()
state.level += 1  # Player beat a level
state.lives -= 1  # Player had to try again


# Example 3
import pickle
state_path = 'game_state.bin'
with open(state_path, 'wb') as f:
    pickle.dump(state, f)


# Example 4
with open(state_path, 'rb') as f:
    state_after = pickle.load(f)
print(state_after.__dict__)


# Example 5
class GameState(object):
    def __init__(self):
        self.level = 0
        self.lives = 4
        self.points = 0


# Example 6
state = GameState()
serialized = pickle.dumps(state)
state_after = pickle.loads(serialized)
print(state_after.__dict__)
