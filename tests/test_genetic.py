from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pytest

from GeneticAlgorithm import TEST_SEEDS, GeneticAlgorithm, get_fitness, play_games
from SnakeGame import MOVES


def make_ga(**kwargs):
    params = dict(size=20, pop_size=20, num_gen=1, window_size=7, hidden_size=8, n_games=1, seed=0)
    params.update(kwargs)
    return GeneticAlgorithm(**params)


def same_indiv(a, b):
    return all(np.array_equal(la, lb) for la, lb in zip(a, b))


def greedy_indiv(ga):
    """Réseau fait à la main : l'entrée « pomme à gauche » active la sortie gauche, etc."""
    indiv = [np.zeros_like(layer) for layer in ga.pop[0]]
    for k in range(len(MOVES)):
        indiv[0][k, ga.window_size**2 + k] = 1
        indiv[1][k, k] = 1
        indiv[2][k, k] = 1
    return indiv


def test_fitness_formula_from_report():
    assert get_fitness(SimpleNamespace(iter=50, score=3)) == 50 * 2**3


def test_fitness_stays_finite_with_a_huge_score():
    # 2.0**score est un flottant : ni entier géant ni infini, même grille pleine (397 pommes en 20 x 20)
    fitness = get_fitness(SimpleNamespace(iter=10**5, score=400))
    assert np.isfinite(fitness) and fitness > 0


def test_play_games_averages_over_seeds():
    ga = make_ga()
    indiv = ga.pop[0]
    f1, a1 = play_games((indiv, [1], 20, 7, 100))
    f2, a2 = play_games((indiv, [2], 20, 7, 100))
    f, a = play_games((indiv, [1, 2], 20, 7, 100))
    assert f == (f1 + f2) / 2 and a == (a1 + a2) / 2


def test_population_size_stays_constant():
    # L'ancienne version passait de 30 à 33 individus après une génération.
    ga = make_ga(pop_size=30)
    for _ in range(2):
        ga.one_generation()
        assert len(ga.pop) == 30


def test_crossover_takes_each_gene_from_a_parent():
    ga = make_ga()
    p1, p2 = ga.pop[0], ga.pop[1]
    child = ga.crossover(p1, p2)
    from_p1 = 0
    for c, l1, l2 in zip(child, p1, p2):
        assert c.shape == l1.shape
        assert np.all((c == l1) | (c == l2))
        from_p1 += (c == l1).sum()
    total = sum(layer.size for layer in p1)
    assert 0.4 < from_p1 / total < 0.6


def test_mutation_rate_and_size():
    ga = make_ga(mutation_chance=0.3, mutation_size=0.2)
    indiv = ga.pop[0]
    before = [layer.copy() for layer in indiv]
    mutant = ga.mutate(indiv)
    assert same_indiv(indiv, before)  # l'original n'est pas modifié
    changed = sum((m != l).sum() for m, l in zip(mutant, indiv))
    total = sum(layer.size for layer in indiv)
    assert 0.25 < changed / total < 0.35
    assert max(np.abs(m - l).max() for m, l in zip(mutant, indiv)) <= 0.2


def test_no_mutation_when_chance_is_zero():
    ga = make_ga(mutation_chance=0.0)
    assert same_indiv(ga.mutate(ga.pop[0]), ga.pop[0])


def test_selection_and_elitism_keep_the_best_unchanged():
    ga = make_ga(pop_size=20)
    old_pop = list(ga.pop)
    fitness = np.arange(20.0)  # l'individu 19 est le meilleur, puis 18, ...
    ga.evaluate = lambda pool=None: (fitness, np.zeros(20))
    stats = ga.one_generation()
    assert stats["best_fitness"] == 19
    assert ga.best is old_pop[19]
    ga.one_generation()  # construit la génération suivante à partir du top rank
    top_5 = [old_pop[i] for i in (19, 18, 17, 16, 15)]
    assert all(new is old for new, old in zip(ga.pop[:5], top_5))
    assert len(ga.pop) == 20


def test_last_generation_is_not_reproduced():
    ga = make_ga(pop_size=8)  # une seule génération
    first_pop = list(ga.pop)
    ga.run()
    assert all(a is b for a, b in zip(ga.pop, first_pop))  # la population évaluée, pas une nouvelle


def test_same_seed_same_evolution():
    a, b = make_ga(seed=42), make_ga(seed=42)
    for _ in range(2):
        assert a.one_generation() == b.one_generation()
    assert all(same_indiv(x, y) for x, y in zip(a.pop, b.pop))
    c = make_ga(seed=43)
    assert not same_indiv(a.pop[-1], c.pop[-1])


def test_multiprocessing_gives_the_same_result():
    serial, parallel = make_ga(seed=1), make_ga(seed=1)
    with Pool(2) as pool:
        assert serial.one_generation() == parallel.one_generation(pool)


@pytest.mark.parametrize(
    "bad",
    [
        dict(pop_size=7),
        dict(window_size=6),
        dict(num_gen=0),
        dict(num_gen=-1),
        dict(n_games=0),
        dict(mutation_chance=-0.1),
        dict(mutation_chance=1.5),
    ],
)
def test_bad_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        make_ga(**bad)


def test_saved_best_is_chosen_on_new_games():
    # Le premier à l'entraînement, un serpent nul mais chanceux, ne doit pas être sauvegardé.
    ga = make_ga(pop_size=8)
    dumb = [np.zeros_like(layer) for layer in ga.pop[0]]  # sorties nulles : toujours tout droit
    greedy = greedy_indiv(ga)
    ga.pop[:2] = [dumb, greedy]
    ga.evaluate = lambda pool=None: (np.array([9.0, 1, 0, 0, 0, 0, 0, 0]), np.zeros(8))
    assert ga.run() is greedy


def test_test_curve_replays_the_best_on_fixed_new_games():
    ga = make_ga()
    stats = ga.one_generation()
    assert stats["test_apples"] == play_games((ga.best, TEST_SEEDS, 20, 7, 100))[1]
    assert min(TEST_SEEDS) >= 2**31  # l'entraînement tire ses graines dans [0, 2^31)
