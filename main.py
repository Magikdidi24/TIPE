from GeneticAlgorithm import *
from SnakeGame import *

size = 20
snake_size = 3

pop_size = 30
num_gen = 500
window_size = 7
hidden_size = 8
max_iter = 100

genetic_algorithm = GeneticAlgorithm(
    size,
    pop_size,
    num_gen,
    window_size,
    hidden_size,
    max_iter=max_iter,
    mutation_chance=0.5,
    mutation_size=0.2,
    display=False,
)

genetic_algorithm.run(eating=True)
