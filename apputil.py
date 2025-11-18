import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    Execute one step of Conway's Game of Life.

    Parameters
    current_board : numpy.ndarray
        A 2D binary array representing the current board state.

    Returns
    numpy.ndarray
        Updated board after applying Conway's Game of Life rules.
    """
    rows, cols = current_board.shape
    updated_board = np.zeros((rows, cols), dtype=int)

    # All 8 neighbor directions
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(rows):
        for c in range(cols):
            live_neighbors = 0

            # Count live neighbors
            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    live_neighbors += current_board[nr, nc]

            cell = current_board[r, c]

            # Apply Conway's rules
            if cell == 1:
                # Dies by loneliness or overcrowding
                if live_neighbors < 2 or live_neighbors > 3:
                    updated_board[r, c] = 0
                else:
                    updated_board[r, c] = 1
            else:
                # Dead cell becomes alive
                if live_neighbors == 3:
                    updated_board[r, c] = 1

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show n_steps of Conway's Game of Life, given update_board().

    Parameters
    game_board : numpy.ndarray
        Binary array representing initial board.
    n_steps : int
        Number of updates to show.
    pause : float
        Seconds to pause between frames.
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # draw board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait before next step
        if step + 1 < n_steps:
            time.sleep(pause)


def play_game_recursive(steps, board=None):
    """
    Recursively play Conway's Game of Life.

    Parameters
    steps : int
        Number of recursive steps.
    board : numpy.ndarray, optional
        Initial board. If None, generate random 10×10

    Returns
    numpy.ndarray
        Final board after recursion finishes.
    """
    if board is None:
        board = np.random.randint(2, size=(10, 10))

    if steps == 0:
        return board

    next_board = update_board(board)
    return play_game_recursive(steps - 1, next_board)


def knapsack(W, weights, values, full_table=False):
    """
    Solve the knapsack problem using dynamic programming.

    Parameters
    W : int
        Maximum weight capacity.
    weights : list[int]
        Weights of each item.
    values : list[int]
        Values of each item.
    full_table : bool, optional
        If True, return the entire DP table.

    Returns
    int or list[list[int]]
        Maximum achievable value, or full DP table.
    """
    # Number of items
    n = len(values)

    # Create DP table (n+1 rows, W+1 columns)
    table = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Fill DP table
    for i in range(n + 1):
        for j in range(W + 1):

            # Base case: no items or zero capacity
            if i == 0 or j == 0:
                table[i][j] = 0

            # If item fits into weight j
            elif weights[i - 1] <= j:
                take_value = values[i - 1] + table[i - 1][j - weights[i - 1]]
                skip_value = table[i - 1][j]
                table[i][j] = max(take_value, skip_value)

            # Item does not fit → skip
            else:
                table[i][j] = table[i - 1][j]

    # Return full table if requested
    if full_table:
        return table

    # Return the best achievable value
    return table[n][W]