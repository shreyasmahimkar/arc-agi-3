import arc_agi
from arcengine import GameAction

def main():
    print("Initializing ARC-AGI-3 Arcade...")
    arc = arc_agi.Arcade()
    
    game_id = "ls20"
    print(f"Loading environment for game: '{game_id}'")
    env = arc.make(game_id, render_mode="terminal")
    
    print("\nAvailable Action Space:")
    print(env.action_space)
    
    print(f"\nTaking ACTION1...")
    obs = env.step(GameAction.ACTION1)
    
    print("\nScorecard Result:")
    print(arc.get_scorecard())

if __name__ == "__main__":
    main()
