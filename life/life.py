"""An Implementation of Conways Game of Life."""
import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d

glider = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])

blinker = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])

glider_gun = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0]
])


class Game:
    """Game class shows the Game of Life board and its operations."""

    def __init__(self, size):
        """
        Initialize the game board with zeros.

        :param size: Size of the game board (size x size).
        """
        self.board = np.zeros((size, size))

    def play(self):
        """
        Start playing the Game of Life.

        Press Ctrl+C to stop.
        """
        print("Playing life. Press Ctrl+C to stop.")
        pyplot.ion()
        while True:
            self.move()
            self.show()
            pyplot.pause(0.0000005)

    def move(self):
        """Move to the next generation based on the Game of Life rules."""
        stencil = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbourcount = convolve2d(self.board, stencil, mode='same')

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 1 if (
                    neighbourcount[i, j] == 3 or (
                        neighbourcount[i, j] == 2 and self.board[i, j])
                ) else 0

    def __setitem__(self, key, value):
        """
        Set a cell's state (alive or dead) at the specified key.

        :param key: Tuple (x, y) representing the cell coordinates.
        :param value: 0 for dead, 1 for alive.
        """
        self.board[key] = value

    def show(self):
        """Display the current game board."""
        pyplot.clf()
        pyplot.matshow(self.board, fignum=0, cmap='binary')
        pyplot.show()

    def insert(self, pattern, location):
        """
        Insert a given Pattern at a specified location on the game board.

        :param pattern: A Pattern object to insert.
        :param location: (x, y) showing the location of Pattern centre.
        """
        pattern_grid = pattern.grid
        pattern_size_x, pattern_size_y = pattern_grid.shape
        x, y = location

        x_start = max(0, x - pattern_size_x // 2)
        x_end = min(self.board.shape[0], x + (pattern_size_x + 1) // 2)
        y_start = max(0, y - pattern_size_y // 2)
        y_end = min(self.board.shape[1], y + (pattern_size_y + 1) // 2)

        pattern_x_start = max(0, pattern_size_x // 2 - x)
        pattern_x_end = min(
            pattern_size_x, self.board.shape[0] - x + pattern_size_x // 2
        )
        pattern_y_start = max(0, pattern_size_y // 2 - y)
        pattern_y_end = min(
            pattern_size_y, self.board.shape[1] - y + pattern_size_y // 2
        )

        self.board[x_start:x_end, y_start:y_end] = pattern_grid[
            pattern_x_start:pattern_x_end, pattern_y_start:pattern_y_end
        ]


class Pattern:
    """Pattern class represents a pattern of cells in the Game of Life."""

    def __init__(self, grid):
        """
        Construct a Pattern class.

        :param grid: A numpy array containing a pattern of 1s and 0s.
        """
        self.grid = grid

    def flip_vertical(self):
        """
        Flip the pattern vertically and return a new Pattern.

        :return: A new Pattern with the pattern flipped upside down.
        """
        flipped_grid = self.grid[::-1]
        return Pattern(flipped_grid)

    def flip_horizontal(self):
        """
        Flip the pattern horizontally and return a new Pattern.

        :return: A new Pattern with the pattern reversed left-right.
        """
        flipped_grid = self.grid[:, ::-1]
        return Pattern(flipped_grid)

    def flip_diag(self):
        """
        Flip the pattern diagonally (transpose) and return a new Pattern.

        :return: A new Pattern with the pattern transposed.
        """
        flipped_grid = np.transpose(self.grid)
        return Pattern(flipped_grid)

    def rotate(self, n):
        """
        Rotate the pattern n right angles anticlockwise to return new Pattern.

        :param n: Number of right angles to rotate (90 degrees per angle).
        :return: A new Pattern with the rotated pattern.
        """
        rotated_grid = np.rot90(self.grid, k=n)
        return Pattern(rotated_grid)
