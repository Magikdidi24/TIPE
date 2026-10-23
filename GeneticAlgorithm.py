from contextlib import nullcontext
from multiprocessing import Pool
from typing import List

import numpy as np

from NeuralNetwork import NeuralPlayer, generate_indiv
from SnakeGame import MOVES, SNAKE_SIZE, SnakeGame

# Réglages par défaut (repris par main.py train)
SIZE = 20  # Taille de la grille (size x size).
POP_SIZE = 200  # Taille de la population (30 dans le rapport : trop peu pour apprendre).
NUM_GEN = 300  # Nombre de générations.
WINDOW_SIZE = 7  # Taille de la fenêtre de vision du serpent.
HIDDEN_SIZE = 8  # Taille des 2 couches cachées du réseau de neurones.
MAX_ITER = 100  # Nombre maximal de déplacements sans manger (famine).
MUTATION_CHANCE = 0.5  # Probabilité de mutation de chaque poids.
MUTATION_SIZE = 0.2  # Amplitude maximale d'une mutation.
N_GAMES = 5  # Parties jouées par chaque serpent (fitness moyenne).

# Graines jamais tirées pendant l'entraînement (qui tire dans [0, 2^31)) : des parties nouvelles
SELECTION_SEEDS = range(2**31, 2**31 + 100)  # choix final du meilleur serpent
TEST_SEEDS = range(2**31 + 100, 2**31 + 130)  # courbe de test : les mêmes 30 parties à chaque génération


def get_fitness(game: SnakeGame):
    """Fitness du rapport : nombre de déplacements x 2^(pommes mangées)."""
    return game.iter * 2.0**game.score


def play_games(args):
    """Fait jouer un individu une partie par graine.

    Renvoie (fitness moyenne, nombre moyen de pommes). Fonction de module pour
    pouvoir être envoyée aux processus de multiprocessing.Pool.
    """
    indiv, seeds, size, window_size, max_iter = args
    fitness, apples = [], []
    for seed in seeds:
        player = NeuralPlayer(indiv, window_size)
        game = SnakeGame(size, SNAKE_SIZE, player, max_iter=max_iter, seed=seed)
        game.run()
        fitness.append(get_fitness(game))
        apples.append(game.score)
    return np.mean(fitness), np.mean(apples)


class GeneticAlgorithm:
    def __init__(
        self,
        size: int = SIZE,
        pop_size: int = POP_SIZE,
        num_gen: int = NUM_GEN,
        window_size: int = WINDOW_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        max_iter=MAX_ITER,
        mutation_chance=MUTATION_CHANCE,
        mutation_size=MUTATION_SIZE,
        n_games=N_GAMES,
        seed=None,
    ):
        if pop_size < 8:
            raise ValueError("pop_size doit valoir au moins 8 (il faut 2 parents dans le top 25 %)")
        if window_size % 2 == 0:
            raise ValueError("window_size doit être impair")
        if num_gen < 1:
            raise ValueError("num_gen doit valoir au moins 1")
        if n_games < 1:
            raise ValueError("n_games doit valoir au moins 1")
        if not 0 <= mutation_chance <= 1:
            raise ValueError("mutation_chance doit être entre 0 et 1")
        self.size = size

        self.pop_size = pop_size
        self.num_gen = num_gen

        self.window_size = window_size
        self.input_size = self.window_size**2 + 4  # vision + direction de la pomme
        self.hidden_size = hidden_size
        self.output_size = len(MOVES)

        self.max_iter = max_iter
        self.n_games = n_games  # parties par individu (on moyenne pour réduire le hasard)

        self.mutation_chance = mutation_chance
        self.mutation_size = mutation_size

        # Un seul générateur aléatoire, initialisé par la graine : résultats reproductibles
        self.rng = np.random.default_rng(seed)

        self.pop = [self.new_indiv() for _ in range(pop_size)]
        self.history = []  # statistiques de chaque génération
        self.best = None  # meilleur individu de la dernière génération (après run : choix final)

    def new_indiv(self):
        return generate_indiv(self.input_size, self.hidden_size, self.output_size, self.rng)

    def evaluate(self, pool=None):
        """Fait jouer toute la population sur les mêmes parties (mêmes graines)."""
        seeds = self.rng.integers(2**31, size=self.n_games).tolist()
        args = [(indiv, seeds, self.size, self.window_size, self.max_iter) for indiv in self.pop]
        results = list((pool.map if pool else map)(play_games, args))
        fitness, apples = np.array(results).T
        return fitness, apples

    def one_generation(self, pool=None):
        if self.history:  # nouvelle population, à partir du top rank de la génération précédente
            self.pop = self.reproduce(self.top_25)
        fitness, apples = self.evaluate(pool)

        order = np.argsort(-fitness, kind="stable")  # du meilleur au moins bon
        self.best = self.pop[order[0]]
        stats = {
            "generation": len(self.history),
            "best_fitness": float(fitness[order[0]]),
            "mean_fitness": float(fitness.mean()),
            "best_apples": float(apples[order[0]]),
            "mean_apples": float(apples.mean()),
            "test_apples": float(self.held_out_apples([self.best], TEST_SEEDS, pool)[0]),
        }
        self.history.append(stats)

        self.top_25 = [self.pop[i] for i in order[: self.pop_size // 4]]
        return stats

    def held_out_apples(self, indivs, seeds, pool=None):
        """Pommes par partie de chaque individu, sur des parties jamais utilisées pour la sélection."""
        args = [
            (indiv, [seed], self.size, self.window_size, self.max_iter) for indiv in indivs for seed in seeds
        ]
        apples = [a for _, a in (pool.map if pool else map)(play_games, args)]
        return np.reshape(apples, (len(indivs), len(seeds))).mean(axis=1)

    def run(self, workers=1):
        """Lance num_gen générations et renvoie le meilleur individu.

        Le premier de la dernière génération est classé sur n_games parties seulement : c'est
        souvent un serpent chanceux. On rejoue donc les 10 premiers sur 100 parties nouvelles
        et on garde celui qui y mange le plus de pommes.
        """
        with Pool(workers) if workers > 1 else nullcontext() as pool:
            for _ in range(self.num_gen):
                s = self.one_generation(pool)
                print(
                    f"Génération {s['generation']:4d} | "
                    f"fitness max {s['best_fitness']:10.4g}  moy {s['mean_fitness']:10.4g} | "
                    f"pommes meilleur {s['best_apples']:5.1f}  moy {s['mean_apples']:5.2f}  "
                    f"test {s['test_apples']:5.2f}",
                    flush=True,
                )
            candidates = self.top_25[:10]
            apples = self.held_out_apples(candidates, SELECTION_SEEDS, pool)
            self.best = candidates[int(np.argmax(apples))]
        print(
            f"Choix final, pommes par partie sur {len(SELECTION_SEEDS)} parties nouvelles : "
            f"premier de la dernière génération {apples[0]:.2f}, serpent gardé {apples.max():.2f}"
        )
        return self.best

    def reproduce(self, top_25: List[List[np.ndarray]]):
        """Generate a new population from the best of the previous one

        Comme dans le rapport, la nouvelle population est formée de :
          1. le top rank (25 % meilleurs), recopié tel quel (élitisme) ;
          2. des enfants : croisement de deux parents du top rank, puis mutation ;
          3. une mutation de chaque individu du top rank ;
          4. quelques individus aléatoires (préservation de la diversité génétique).

        Args:
            top_25 (List[List[np.ndarray]]): the best individuals, best first

        Returns:
            [List[List[np.ndarray]]]: the new population (pop_size individuals)
        """
        n_random = self.pop_size // 10
        n_children = self.pop_size - 2 * len(top_25) - n_random

        new_pop = list(top_25)
        for _ in range(n_children):
            i, j = self.rng.choice(len(top_25), 2, replace=False)
            new_pop.append(self.mutate(self.crossover(top_25[i], top_25[j])))
        new_pop += [self.mutate(indiv) for indiv in top_25]
        new_pop += [self.new_indiv() for _ in range(n_random)]
        return new_pop

    def crossover(self, parent1, parent2):
        """Croisement uniforme : chaque poids vient de l'un ou l'autre parent (probabilité 1/2)."""
        return [
            np.where(self.rng.random(layer1.shape) < 0.5, layer1, layer2)
            for layer1, layer2 in zip(parent1, parent2)
        ]

    def mutate(self, indiv):
        """Mutation de point : chaque poids est modifié avec la probabilité mutation_chance,
        d'une valeur uniforme(-1, 1) x mutation_size."""
        new_indiv = []
        for layer in indiv:
            mask = self.rng.random(layer.shape) < self.mutation_chance
            noise = self.rng.uniform(-1, 1, layer.shape) * self.mutation_size
            new_indiv.append(layer + mask * noise)
        return new_indiv
