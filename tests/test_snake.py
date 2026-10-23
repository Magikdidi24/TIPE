import pytest

from SnakeGame import DOWN, LEFT, RIGHT, UP, RandomPlayer, SnakeGame


class ScriptedPlayer:
    """Joueur qui rejoue une liste de coups (en boucle)."""

    def __init__(self, moves):
        self.moves = moves
        self.i = 0

    def get_action(self, snake, grid):
        move = self.moves[self.i % len(self.moves)]
        self.i += 1
        return move


def put_apple(game, cell):
    """Déplace la pomme sur une case choisie (pour rendre un test déterministe)."""
    game.grid[game.apple] = 0
    game.apple = cell
    game.grid[cell] = 2


def test_initial_state():
    game = SnakeGame(20, 3, ScriptedPlayer([RIGHT]), seed=0)
    assert game.snake == [(10, 10), (10, 11), (10, 12)]
    assert all(game.grid[c] == 1 for c in game.snake)
    assert game.grid[game.apple] == 2
    assert game.apple not in game.snake
    assert (game.grid == 2).sum() == 1


def test_move_keeps_length_and_clears_tail():
    game = SnakeGame(20, 3, ScriptedPlayer([UP]), seed=0)
    put_apple(game, (0, 0))
    game.move_snake()
    assert game.snake == [(10, 11), (10, 12), (9, 12)]
    assert game.grid[10, 10] == 0
    assert game.grid[9, 12] == 1
    assert not game.crashed


def test_reverse_move_is_ignored():
    # Comme dans la version C : un demi-tour est ignoré, le serpent continue tout droit.
    game = SnakeGame(20, 3, ScriptedPlayer([LEFT]), seed=0)
    put_apple(game, (0, 0))
    game.move_snake()
    assert game.snake[-1] == (10, 13)
    assert not game.crashed


def test_eating_grows_snake_and_respawns_apple():
    game = SnakeGame(20, 3, ScriptedPlayer([RIGHT]), seed=0)
    put_apple(game, (10, 13))
    game.iter_without_apple = 42
    game.move_snake()
    assert game.score == 1
    assert len(game.snake) == 4
    assert game.snake[0] == (10, 10)  # la queue n'a pas bougé
    assert game.iter_without_apple == 0
    assert game.apple not in game.snake
    assert game.grid[game.apple] == 2


def test_wall_collision():
    game = SnakeGame(20, 3, ScriptedPlayer([RIGHT]), seed=0)
    put_apple(game, (0, 0))
    for _ in range(7):  # la tête va de la colonne 12 à 19
        game.move_snake()
    assert not game.crashed
    game.move_snake()
    assert game.crashed


def test_self_collision():
    game = SnakeGame(20, 5, ScriptedPlayer([UP, LEFT, DOWN]), seed=0)
    put_apple(game, (0, 0))
    game.move_snake()
    game.move_snake()
    assert not game.crashed
    game.move_snake()  # la tête retombe sur son propre corps
    assert game.crashed


def test_moving_into_tail_cell_is_allowed():
    # La queue avance en même temps que la tête : la case qu'elle libère est libre.
    game = SnakeGame(20, 4, ScriptedPlayer([UP, LEFT, DOWN, RIGHT]), seed=0)
    put_apple(game, (0, 0))
    for _ in range(12):
        game.move_snake()
    assert not game.crashed


def test_starvation_ends_a_looping_snake():
    game = SnakeGame(20, 3, ScriptedPlayer([UP, LEFT, DOWN, RIGHT]), max_iter=50, seed=0)
    put_apple(game, (0, 0))
    score = game.run()
    assert score == 0
    assert not game.crashed
    assert game.iter == 50


def test_apple_never_on_snake_and_grid_consistent():
    for seed in range(20):
        game = SnakeGame(6, 3, RandomPlayer(seed), max_iter=200, seed=seed)
        while not game.crashed and game.iter_without_apple < game.max_iter:
            game.move_snake()
            if game.crashed:
                break
            assert game.apple not in game.snake
            assert (game.grid == 1).sum() == len(game.snake) == len(set(game.snake))
            assert game.grid[game.apple] == 2


def test_apple_goes_to_last_free_cell_then_none():
    game = SnakeGame(6, 3, ScriptedPlayer([RIGHT]), seed=0)
    game.grid[:] = 1
    game.grid[5, 0] = 0
    assert game.generate_apple() == (5, 0)
    game.grid[5, 0] = 1
    assert game.generate_apple() is None


def test_same_seed_same_apples():
    a = SnakeGame(20, 3, ScriptedPlayer([RIGHT]), seed=7)
    b = SnakeGame(20, 3, ScriptedPlayer([RIGHT]), seed=7)
    assert [a.generate_apple() for _ in range(10)] == [b.generate_apple() for _ in range(10)]


def test_random_player_never_turns_back():
    player = RandomPlayer(0)
    snake = [(10, 10), (10, 11), (10, 12)]  # va vers la droite
    assert {player.get_action(snake, None) for _ in range(100)} == {RIGHT, UP, DOWN}


def test_grid_too_small_for_the_starting_snake_is_rejected():
    SnakeGame(5, 3, ScriptedPlayer([RIGHT]), seed=0)  # colonnes 2, 3, 4 : tient juste
    with pytest.raises(ValueError):
        SnakeGame(4, 3, ScriptedPlayer([RIGHT]), seed=0)
