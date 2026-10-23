import json
import sys

import numpy as np
import pytest

import main
from NeuralNetwork import NeuralPlayer, generate_indiv
from SnakeGame import SNAKE_SIZE, SnakeGame


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *map(str, argv)])
    main.main()


def test_train_then_play_then_plot(tmp_path, monkeypatch, capsys):
    out = tmp_path / "res"
    run_cli(monkeypatch, "train", "--pop-size", 8, "--generations", 2, "--games", 1,
            "--workers", 1, "--seed", 3, "--out", out)
    history = json.loads((out / "history.json").read_text())
    assert len(history["history"]) == 2
    assert history["params"]["seed"] == 3
    assert "test_apples" in history["history"][-1]

    run_cli(monkeypatch, "play", "--genome", out / "best.npz", "--games", 2, "--seed", 0, "--no-display")
    assert "Moyenne" in capsys.readouterr().out

    random_play = ["play", "--genome", out / "best.npz", "--games", 3, "--seed", 0, "--no-display", "--random"]
    run_cli(monkeypatch, *random_play)
    first = capsys.readouterr().out
    run_cli(monkeypatch, *random_play)
    assert "Moyenne" in first and capsys.readouterr().out == first  # mêmes parties, mêmes coups

    run_cli(monkeypatch, "plot", "--history", out / "history.json", "--out", out / "new" / "c.png")
    assert (out / "new" / "c.png").stat().st_size > 0  # dossier créé au besoin


def test_play_zero_games_is_rejected(monkeypatch):
    with pytest.raises(ValueError):
        run_cli(monkeypatch, "play", "--games", 0, "--no-display")


def test_save_then_load_gives_the_same_snake(tmp_path):
    indiv = generate_indiv(7**2 + 4, 8, 4, np.random.default_rng(0))
    main.save_genome(tmp_path / "g.npz", indiv, 20, 7, 100)
    loaded, size, window, iters = main.load_genome(tmp_path / "g.npz")
    assert (size, window, iters) == (20, 7, 100)
    assert len(loaded) == len(indiv) and all(np.array_equal(a, b) for a, b in zip(loaded, indiv))

    def moves(genome, seed):
        game = SnakeGame(size, SNAKE_SIZE, NeuralPlayer(genome, window), max_iter=iters, seed=seed)
        played = []
        while not game.is_over():
            played.append(game.move_snake())
        return played

    assert all(moves(loaded, seed) == moves(indiv, seed) for seed in range(10))
