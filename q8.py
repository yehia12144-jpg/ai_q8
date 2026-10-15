"""
8-Puzzle Solver using A* Search Algorithm
KH5001CMD / KH5002CMD - Task 8: Advanced Heuristic Design
"""

import time
import heapq
from typing import List, Tuple, Dict, Callable

# Goal State representation (0 represents the blank space)
# Layout:
# 1 2 3
# 4 5 6
# 7 8 0
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POSITIONS = {tile: (i // 3, i % 3) for i, tile in enumerate(GOAL_STATE) if tile != 0}

# Precomputed legal moves for the blank tile (indices 0 to 8)
MOVES = {
    0: [1, 3],       # right, down
    1: [0, 2, 4],    # left, right, down
    2: [1, 5],       # left, down
    3: [0, 4, 6],    # up, right, down
    4: [1, 3, 5, 7], # up, left, right, down
    5: [2, 4, 8],    # up, left, down
    6: [3, 7],       # up, right
    7: [4, 6, 8],    # up, left, right
    8: [5, 7]        # up, left
}

# --- 1. SOLVABILITY CHECK ---

def count_inversions(state):
    """
    Counts tile inversions (excluding blank 0).
    An inversion is a pair (a, b) where a appears before b, but a > b.
    """
    tiles = [tile for tile in state if tile != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions

def is_solvable(state):
    """
    In a 3x3 8-puzzle with the blank at the bottom-right in the goal state,
    a state is reachable/solvable if and only if the number of inversions is EVEN.
    """
    return count_inversions(state) % 2 == 0

# --- 2. HEURISTICS ---

def h_misplaced_tiles(state: Tuple):
    """
    Heuristic 1: Misplaced Tiles (Hamming Distance)
    Counts numbered tiles that are not in their goal square.
    """
    count = 0
    for i in range(9):
        tile = state[i]
        if tile != 0 and tile != GOAL_STATE[i]:
            count += 1
    return count

def h_manhattan_distance(state): 
    """
    Heuristic 2: Manhattan Distance (Taxicab Distance)
    Sums the horizontal and vertical grid moves each tile needs to reach home.
    """
    total_dist = 0
    for i in range(9):
        tile = state[i]
        if tile != 0:
            current_r, current_c = i // 3, i % 3
            goal_r, goal_c = GOAL_POSITIONS[tile]
            total_dist += abs(current_r - goal_r) + abs(current_c - goal_c)
    return total_dist

def h_linear_conflict(state): 
    """
    Heuristic 3: Linear Conflict + Manhattan Distance
    Adds +2 moves for any two tiles in their goal row/column that block each other.
    """
    manhattan = h_manhattan_distance(state)
    conflicts = 0

    # Row linear conflicts
    for r in range(3):
        row_tiles = []
        for c in range(3):
            tile = state[r * 3 + c]
            if tile != 0:
                goal_r, goal_c = GOAL_POSITIONS[tile]
                if goal_r == r:
                    row_tiles.append((c, goal_c, tile))
        
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                c_i, goal_c_i, _ = row_tiles[i]
                c_j, goal_c_j, _ = row_tiles[j]
                if goal_c_i > goal_c_j:
                    conflicts += 1

    # Column linear conflicts
    for c in range(3):
        col_tiles = []
        for r in range(3):
            tile = state[r * 3 + c]
            if tile != 0:
                goal_r, goal_c = GOAL_POSITIONS[tile]
                if goal_c == c:
                    col_tiles.append((r, goal_r, tile))
        
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                r_i, goal_r_i, _ = col_tiles[i]
                r_j, goal_r_j, _ = col_tiles[j]
                if goal_r_i > goal_r_j:
                    conflicts += 1

    return manhattan + 2 * conflicts

# --- 3. A* SEARCH ALGORITHM ---

def get_neighbors(state):
    """Generates valid next boards by sliding an adjacent tile into the blank."""
    blank_idx = state.index(0)
    neighbors = []
    state_list = list(state)
    
    for swap_idx in MOVES[blank_idx]:
        new_state = list(state_list)
        new_state[blank_idx], new_state[swap_idx] = new_state[swap_idx], new_state[blank_idx]
        neighbors.append(tuple(new_state))
        
    return neighbors

def a_star_search(initial_state, heuristic):
    """
    Executes A* search using f(n) = g(n) + h(n).
    Returns: (is_solved, path_length, nodes_expanded, elapsed_time_sec)
    """
    if not is_solvable(initial_state):
        return False, -1, 0, 0.0

    start_time = time.perf_counter()
    
    counter = 0
    h_init = heuristic(initial_state)
    # Priority queue entry: (f_score, h_score, tie_breaker, state)
    open_heap = [(h_init, h_init, counter, initial_state)]
    
    g_score: Dict[Tuple[int, ...], int] = {initial_state: 0}
    closed_set = set()
    nodes_expanded = 0

    while open_heap:
        f, h, _, current = heapq.heappop(open_heap)
        
        if current in closed_set:
            continue
            
        if current == GOAL_STATE:
            elapsed_time = time.perf_counter() - start_time
            return True, g_score[current], nodes_expanded, elapsed_time
            
        closed_set.add(current)
        nodes_expanded += 1

        current_g = g_score[current]
        
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
                
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                h_val = heuristic(neighbor)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(open_heap, (f_val, h_val, counter, neighbor))

    elapsed_time = time.perf_counter() - start_time
    return False, -1, nodes_expanded, elapsed_time

def format_grid(state: Tuple[int, ...]) -> str:
    """Formats a state tuple as a 3x3 grid."""
    lines = []
    for r in range(3):
        row_vals = [" " if state[r * 3 + c] == 0 else str(state[r * 3 + c]) for c in range(3)]
        lines.append("  [" + "][".join(row_vals) + "]")
    return "\n".join(lines)

if __name__ == "__main__":
    test_states = [
        {
            "name": "State 1 (Easy / Shallow)",
            "state": (0, 1, 2, 4, 5, 3, 7, 8, 6),
            "description": "Shallow solvable configuration (Optimal depth = 4)"
        },
        {
            "name": "State 2 (Medium / Moderate)",
            "state": (5, 4, 2, 7, 1, 3, 0, 8, 6),
            "description": "Moderate solvable configuration (Optimal depth = 10)"
        },
        {
            "name": "State 3 (Hard / Deep)",
            "state": (7, 2, 4, 5, 0, 6, 8, 3, 1),
            "description": "Deep solvable configuration (Optimal depth = 20)"
        },
        {
            "name": "State 4 (Unsolvable - Adjacent Swap)",
            "state": (1, 2, 3, 4, 5, 6, 8, 7, 0),
            "description": "Goal state with 7 and 8 swapped (1 inversion, odd)"
        },
        {
            "name": "State 5 (Unsolvable - High Inversions)",
            "state": (8, 1, 2, 0, 4, 3, 7, 6, 5),
            "description": "Scrambled state with 11 inversions (odd)"
        }
    ]

    heuristics = [
        ("h1: Misplaced Tiles", h_misplaced_tiles),
        ("h2: Manhattan Distance", h_manhattan_distance),
        ("h3: Linear Conflict", h_linear_conflict)
    ]

    print("=" * 80)
    print("8-PUZZLE A* SEARCH BENCHMARK RESULTS")
    print("=" * 80)

    for test in test_states:
        name = test["name"]
        st = test["state"]
        solvable = is_solvable(st)
        inv = count_inversions(st)
        
        print(f"\n### {name}")
        print(f"Board:\n{format_grid(st)}")
        print(f"Inversions: {inv} | Solvable: {'Yes' if solvable else 'No (Odd Inversions)'}")
        
        if not solvable:
            print("Status: Unsolvable state detected immediately by inversion parity test.")
            print("-" * 80)
            continue
            
        print(f"{'Heuristic':<25} | {'Solved?':<8} | {'Path Length':<12} | {'Nodes Expanded':<16} | {'Time (ms)':<10}")
        print("-" * 80)
        
        for h_name, h_func in heuristics:
            solved, depth, nodes, t_sec = a_star_search(st, h_func)
            t_ms = t_sec * 1000.0
            print(f"{h_name:<25} | {str(solved):<8} | {depth:<12} | {nodes:<16} | {t_ms:<10.2f}")
        print("-" * 80)