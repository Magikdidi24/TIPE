import numpy as np

from NeuralNetwork import NeuralPlayer, forward, generate_indiv
from SnakeGame import APPLE, DOWN, LEFT, MOVES, RIGHT, SNAKE, UP


def test_genome_shapes_and_range():
    indiv = generate_indiv(49, 8, 4, np.random.default_rng(0))
    assert [layer.shape for layer in indiv] == [(8, 50), (8, 9), (4, 9)]
    assert all(np.all(np.abs(layer) <= 1) for layer in indiv)


def test_forward_shape_range_and_determinism():
    indiv = generate_indiv(49, 8, 4, np.random.default_rng(0))
    x = np.random.default_rng(1).integers(-1, 2, 49)
    out = forward(indiv, x)
    assert out.shape == (4,)
    assert np.all(np.abs(out) < 1)  # tanh
    assert np.array_equal(out, forward(indiv, x))


def test_vision_window_contents():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = SNAKE  # tête
    grid[10, 11] = SNAKE
    grid[8, 9] = APPLE
    player = NeuralPlayer(None, 3)
    vision = player.visualize_grid(grid, 10, 10).reshape(3, 3)
    assert vision[1, 1] == -1 and vision[1, 2] == -1
    assert (vision == 0).sum() == 7

    vision = NeuralPlayer(None, 5).visualize_grid(grid, 10, 10).reshape(5, 5)
    assert vision[0, 1] == 1  # la pomme 2 lignes au-dessus, 1 colonne à gauche


def test_vision_sees_walls_as_obstacles():
    grid = np.zeros((20, 20), dtype=int)
    grid[0, 0] = SNAKE
    vision = NeuralPlayer(None, 7).visualize_grid(grid, 0, 0).reshape(7, 7)
    assert np.all(vision[:3, :] == -1) and np.all(vision[:, :3] == -1)
    assert np.all(vision[4:, 4:] == 0)


def test_apple_direction_follows_moves_order():
    grid = np.zeros((20, 20), dtype=int)
    grid[3, 15] = APPLE  # en haut à droite de la tête (10, 10)
    direction = NeuralPlayer(None, 7).apple_direction(grid, 10, 10)
    assert dict(zip(MOVES, direction)) == {LEFT: 0, RIGHT: 1, UP: 1, DOWN: 0}


def test_get_action_returns_a_move():
    indiv = generate_indiv(49 + 4, 8, 4, np.random.default_rng(0))
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = SNAKE
    grid[0, 0] = APPLE
    move = NeuralPlayer(indiv, 7).get_action([(10, 10)], grid)
    assert move in MOVES
