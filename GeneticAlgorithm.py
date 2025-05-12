import math
import random

import numpy as np

from typing import List, Tuple
from SnakeGame import SnakeGame

LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)

MOVES = [LEFT, RIGHT, UP, DOWN]


class GeneticAlgorithm(object):
    def __init__(
        self,
        size: int,
        pop_size: int,
        num_gen: int,
        window_size: int,
        hidden_size: int,
        max_iter=100,
        mutation_chance=0.1,
        mutation_size=0.1,
        display=False,
    ):
        self.size = size

        self.pop_size = pop_size
        self.num_gen = num_gen

        self.window_size = window_size
        self.input_size = self.window_size ** 2
        self.hidden_size = hidden_size

        self.max_iter = max_iter

        self.display = display

        self.output_size = len(MOVES)

        self.mutation_chance = mutation_chance
        self.mutation_size = mutation_size

        self.current_individual = None

        self.pop = [
            self.generate_indiv(self.input_size, self.hidden_size, self.output_size)
            for _ in range(pop_size)
        ]

    def generate_indiv(self, input_size: int, hidden_size: int, output_size: int):
        hidden_layer1 = np.array(
            [
                [random.uniform(-1, 1) for _ in range(input_size + 1)]
                for _ in range(hidden_size)
            ]
        )
        hidden_layer2 = np.array(
            [
                [random.uniform(-1, 1) for _ in range(hidden_size + 1)]
                for _ in range(hidden_size)
            ]
        )
        output_layer = np.array(
            [
                [random.uniform(-1, 1) for _ in range(hidden_size + 1)]
                for _ in range(output_size)
            ]
        )
        return [hidden_layer1, hidden_layer2, output_layer]

    def vizualize_grid(self, grid: np.array, y_head: int, x_head: int):
        # Initialize vision
        snake_vision = [
            [0 for _ in range(self.window_size)] for _ in range(self.window_size)
        ]

        # Compute vision from actual grid
        for i in range(self.window_size):
            ii = y_head - self.window_size // 2 + i
            for j in range(self.window_size):
                jj = x_head - self.window_size // 2 + j
                if (
                    ii < 0 or jj < 0 or ii >= self.size or jj >= self.size# attention à pas dépasser le tableau
                ):  # there is a part of his body
                    snake_vision[i][j] = -1
                elif grid[ii][jj] == 2:  # there is food
                    snake_vision[i][j] = 1
                elif grid[ii][jj] == 0:
                    snake_vision[i][j] = 0
                else:
                    snake_vision[i][j] = -1

        # Flatten the matrix into a vector as the neural network takes a vector as an input
        input_vector = list(np.array(snake_vision).flatten()) + [1]
        input_vector = np.array(input_vector)

        return input_vector

    def get_action(self, snake: List[Tuple[int, int]], grid: np.array):
        input_vector = self.vizualize_grid(grid, snake[-1][0], snake[-1][1])
        hidden_layer1 = self.current_individual[0]
        hidden_layer2 = self.current_individual[1]
        output_layer = self.current_individual[2]

        # Forward propagation
        hidden_result1 = np.array(
            [
                math.tanh(np.dot(input_vector, hidden_layer1[i])) # tangente hyperbolique ?
                for i in range(hidden_layer1.shape[0])
            ]
            + [1]
        )
        hidden_result2 = np.array(
            [
                math.tanh(np.dot(hidden_result1, hidden_layer2[i]))
                for i in range(hidden_layer2.shape[0])
            ]
            + [1]
        )
        output_result = np.array(
            [
                math.tanh(np.dot(hidden_result2, output_layer[i]))
                for i in range(output_layer.shape[0])
            ]
        )

        max_index = np.argmax(output_result)

        return MOVES[max_index]

    def get_fitness(self, game: SnakeGame):
        score = len(game.snake) - game.snake_size
        return game.iter * 2 ** score

    def one_generation(self, eating=False):
        scores = [0 for _ in range(self.pop_size)]
        max_score = 0

        for i in range(self.pop_size):

            if self.display:
                print(f"Snake {i} est en cours...")

            self.current_individual = self.pop[i]

            game = SnakeGame(
                self.size, self.size // 3, self, eating=eating, max_iter=self.max_iter
            )
            outcome = game.run()
            if eating:
                score = self.get_fitness(game)
            else:
                score = game.iter  # ACTUALISATION

            scores[i] += score

            if self.display:
                print(f"Score: {score}\n")
                if outcome == 0:
                    print(f"Snake {i} en vie !\n")

            if score > max_score:
                max_score = score
                if self.display:
                    print(f"Max score: {max_score}, at ID {i}\n")

        top_25_indexes = list(np.argsort(scores))[
            3 * (self.pop_size // 4) : self.pop_size
        ]

        print(f"Scores : {scores}")

        top_25 = [self.pop[i] for i in top_25_indexes][::-1]
        self.pop = self.reproduce(top_25)

    def run(self, eating=False):
        num_gen = self.num_gen if eating else self.num_gen

        for i in range(num_gen):
            self.one_generation(eating)
            print(f"---------------------- Generation {i+1}")

            if i % 100 == 0 or (i == num_gen - 1):
                print("Displaying the top 3 individuals:")
                for indiv in self.pop[:3]:
                    self.current_individual = indiv
                    game = SnakeGame(
                        self.size,
                        self.size // 3,
                        self,
                        eating=eating,
                        max_iter=self.max_iter,
                        display=True,
                    )
                    outcome = game.run(display=True)

        input("Press a key to display the top 3")
        for indiv in self.pop[:3]:
            self.current_individual = indiv
            game = SnakeGame(
                self.size,
                self.size // 3,
                self,
                eating=eating,
                max_iter=self.max_iter,
                display=True,
            )
            outcome = game.run(display=True)

    def reproduce(self, top_25: List[np.array]):
        """Generate a new population from the best of the previous one

        Args:
            top_25 (List[np.array]): Represent a list of the top 25 individuals of the generation

        Returns:
            [List[np.array]]: Return the new population
        """
        new_pop = []
        for indiv in top_25:
            new_pop.append(indiv)
        for indiv in top_25:
            new_indiv = self.mutate(indiv)
            new_pop.append(new_indiv)
        for _ in range(self.pop_size // 2):
            new_pop.append(
                self.generate_indiv(self.input_size, self.hidden_size, self.output_size)
            )
        return new_pop

    def mutate(self, indiv: np.array):
        """Randomly change weights in an individual

        Args:
            indiv (np.array): The actual individual to mutate

        Returns:
            [np.array]: Return the mutated individual
        """
        new_indiv = []
        for layer in indiv:
            new_layer = np.copy(layer)
            for i in range(new_layer.shape[0]):
                for j in range(new_layer.shape[1]):
                    rand = random.uniform(0, 1)
                    if rand <= self.mutation_chance:
                        new_layer[i][j] += random.uniform(-1, 1) * self.mutation_size
            new_indiv.append(new_layer)
        return new_indiv
