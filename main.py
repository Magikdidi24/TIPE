"""TIPE : un Snake qui apprend à jouer grâce à un algorithme génétique.

    python main.py train   # entraîne sans affichage, sauvegarde le meilleur serpent + l'historique
    python main.py play    # regarde le meilleur serpent jouer (fenêtre pygame)
    python main.py plot    # courbes d'apprentissage (image PNG)
"""
import argparse
import json
import os

import numpy as np

from GeneticAlgorithm import (
    HIDDEN_SIZE,
    MAX_ITER,
    MUTATION_CHANCE,
    MUTATION_SIZE,
    N_GAMES,
    NUM_GEN,
    POP_SIZE,
    SIZE,
    WINDOW_SIZE,
    GeneticAlgorithm,
)
from NeuralNetwork import NeuralPlayer
from SnakeGame import SNAKE_SIZE, RandomPlayer, SnakeGame


def save_genome(path, indiv, size, window_size, max_iter):
    """Sauvegarde les couches du réseau (arr_0, arr_1, ...) et les réglages du jeu."""
    np.savez(path, *indiv, size=size, window_size=window_size, max_iter=max_iter)


def load_genome(path):
    """Renvoie (couches du réseau, size, window_size, max_iter) écrits par save_genome."""
    with np.load(path) as data:
        n_layers = sum(name.startswith("arr_") for name in data.files)
        indiv = [data[f"arr_{i}"] for i in range(n_layers)]
        return indiv, int(data["size"]), int(data["window_size"]), int(data["max_iter"])


def train(args):
    os.makedirs(args.out, exist_ok=True)
    genetic_algorithm = GeneticAlgorithm(
        args.size,
        args.pop_size,
        args.generations,
        args.window_size,
        args.hidden_size,
        max_iter=args.max_iter,
        mutation_chance=args.mutation_chance,
        mutation_size=args.mutation_size,
        n_games=args.games,
        seed=args.seed,
    )
    best = genetic_algorithm.run(workers=args.workers)

    genome_path = os.path.join(args.out, "best.npz")
    save_genome(genome_path, best, args.size, args.window_size, args.max_iter)
    history_path = os.path.join(args.out, "history.json")
    with open(history_path, "w") as f:
        params = {k: v for k, v in vars(args).items() if k != "func"}
        json.dump({"params": params, "history": genetic_algorithm.history}, f, indent=1)
    print(f"Meilleur serpent : {genome_path}\nHistorique : {history_path}")


def play(args):
    if args.games < 1:
        raise ValueError("--games doit valoir au moins 1")
    indiv, size, window, iters = load_genome(args.genome)

    scores = []
    for k in range(args.games):
        seed = None if args.seed is None else args.seed + k
        if args.random:  # graine [seed, 1] : un hasard différent de celui des pommes (graine seed)
            player = RandomPlayer(None if seed is None else [seed, 1])
        else:
            player = NeuralPlayer(indiv, window)
        game = SnakeGame(size, SNAKE_SIZE, player, max_iter=iters, seed=seed)
        scores.append(game.run(display=not args.no_display, fps=args.fps))
        end = "mur/corps" if game.crashed else ("famine" if game.apple is not None else "grille pleine")
        print(f"Partie {k + 1} : {game.score} pommes en {game.iter} déplacements (fin : {end})")
    print(f"Moyenne : {np.mean(scores):.2f} pommes, max : {max(scores)}")


def plot(args):
    import matplotlib.pyplot as plt

    with open(args.history) as f:
        history = json.load(f)["history"]
    gen = [h["generation"] for h in history]
    best, mean, test = "#2a78d6", "#eb6834", "#1baf7a"
    train_best = "meilleur serpent (parties d'entraînement)"
    train_mean = "moyenne de la population (parties d'entraînement)"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, facecolor="#fcfcfb")
    ax1.plot(gen, [h["best_fitness"] for h in history], color=best, lw=1.5, label=train_best)
    ax1.plot(gen, [h["mean_fitness"] for h in history], color=mean, lw=1.5, label=train_mean)
    ax1.set_yscale("log")
    ax1.set_ylabel("fitness = déplacements x 2^pommes")
    ax2.plot(gen, [h["best_apples"] for h in history], color=best, lw=1.5, label=train_best)
    ax2.plot(gen, [h["mean_apples"] for h in history], color=mean, lw=1.5, label=train_mean)
    ax2.plot(gen, [h["test_apples"] for h in history], color=test, lw=1.5, label="meilleur serpent, 30 parties de test")
    ax2.set_ylabel("pommes mangées (moyenne par partie)")
    ax2.set_xlabel("génération")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    ax1.set_title("Apprentissage du Snake par algorithme génétique", color="#0b0b0b")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Courbes : {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", help="entraîner les serpents (sans affichage)")
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--pop-size", type=int, default=POP_SIZE)
    p.add_argument("--generations", type=int, default=NUM_GEN)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--mutation-chance", type=float, default=MUTATION_CHANCE)
    p.add_argument("--mutation-size", type=float, default=MUTATION_SIZE)
    p.add_argument("--games", type=int, default=N_GAMES, help="parties par serpent et par génération")
    p.add_argument("--seed", type=int, default=0, help="graine aléatoire (reproductibilité)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processus en parallèle")
    p.add_argument("--out", default="results", help="dossier de sortie")
    p.set_defaults(func=train)

    p = commands.add_parser("play", help="regarder un serpent sauvegardé jouer")
    p.add_argument("--genome", default="results/best.npz")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="graine des pommes (défaut : au hasard)")
    p.add_argument("--fps", type=int, default=10, help="vitesse d'affichage")
    p.add_argument("--no-display", action="store_true", help="sans fenêtre, affiche seulement les scores")
    p.add_argument("--random", action="store_true", help="déplacements au hasard (référence), mêmes parties")
    p.set_defaults(func=play)

    p = commands.add_parser("plot", help="tracer les courbes d'apprentissage")
    p.add_argument("--history", default="results/history.json")
    p.add_argument("--out", default="results/apprentissage.png")
    p.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
