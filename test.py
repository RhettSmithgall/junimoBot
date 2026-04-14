import random
import os
import time

# Clear screen (works on Windows/Linux/Mac)
os.system('cls' if os.name == 'nt' else 'clear')

# Players
players = {
    "Redwick": "R",
    "Gragas": "G",
    "Coven": "C",
    "Silver": "S",
    "Jorkin": "J"
}

# Grid size
WIDTH = 10
HEIGHT = 8

# Random positions
positions = {}
for name in players:
    positions[name] = (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))

# Build grid
grid = [["." for _ in range(WIDTH)] for _ in range(HEIGHT)]

for name, (x, y) in positions.items():
    grid[y][x] = players[name]

# Fake d20 roll
def d20():
    return random.randint(1, 20)

# Chat log
chat = [
    f"Dungeon Master: Welcome, adventurers...",
    f"Redwick rolls a d20... ({d20()})",
    f"Gragas attempts to smash a barrel... ({d20()})",
    f"Coven casts a mysterious spell... ({d20()})",
    f"Silver sneaks into the shadows... ({d20()})",
    f"Jorkin tries something stupid... ({d20()})",
    f"Dungeon Master: The dungeon trembles..."
]

# UI rendering
print("="*50)
print("         DUNGEON TERMINAL INTERFACE ")
print("="*50)

# Grid
print("\nMAP:")
for row in grid:
    print(" ".join(row))

# Chat window
print("\nCHAT:")
print("-"*50)
for line in chat:
    print(line)
print("-"*50)

print("\n> Type 'roll' to roll a d20")

# Optional little animation for flavor
time.sleep(0.5)
print("\nDungeon Master is thinking...")
time.sleep(1)
print("A goblin appears!")


