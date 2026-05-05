import sys
import json

def simulate_step(grid, action):
    # Dummy logic
    return grid, 1.0, False

class LocalSimulator:
    def __init__(self):
        self.grid = [[0 for _ in range(10)] for _ in range(10)]
        
    def step(self, state, action):
        new_grid, reward, done = simulate_step(self.grid, 0)
        return {"grid": new_grid}, reward, done

if __name__ == "__main__":
    if len(sys.argv) > 1:
        memory_file = sys.argv[1]
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
                print("Loaded memory:", len(data))
        except Exception as e:
            pass
    sim = LocalSimulator()
    print("Success")