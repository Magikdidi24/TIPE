import random
import time

import numpy as np
import pygame

SCREEN_SIZE = 600
COLORS = {"white": (255, 255, 255), "green": (0, 255, 0), "red": (255, 0, 0)}

class SnakeGame(object):
    def __init__(
        self,
        size: int,
        snake_size: int,
        player,
        eating=False,
        max_iter=100,
        display=False,
    ):
        self.size = size
        self.snake_size = snake_size
        self.player = player
        self.max_iter = max_iter

        self.iter = 0
        self.crashed = False

        self.grid = np.zeros([self.size, self.size])

        self.snake = [
            (self.size // 2, self.size // 2 + i) for i in range(self.snake_size)
        ]
        for coord in self.snake:
            self.grid[coord[0]][coord[1]] = 1

        self.eating = eating
        self.display = display

        if self.eating:
            self.apples = [(7, 7), (3, 7), (12, 0), (18, 3), (5, 7)]
            # self.apples = [self.generate_apple() for _ in range(20)]
            for apple in self.apples:
                self.grid[apple[0]][apple[1]] = 2
            # self.grid[self.apples[0][0]][self.apples[0][1]] = 2

    def generate_apple(self):
        return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))

    def move_snake(self):
        move = self.player.get_action(self.snake, self.grid)

        new_head = (self.snake[-1][0] + move[0], self.snake[-1][1] + move[1])
        self.snake.append(new_head)

        head = self.snake[-1]

        if self.eating:
            if head not in self.apples:
                # Delete the tail
                self.grid[self.snake[0][0]][self.snake[0][1]] = 0
                self.snake.pop(0)
            else:
                self.apples.pop(self.apples.index(head))
        else:
            # Delete the tail
            self.grid[self.snake[0][0]][self.snake[0][1]] = 0
            self.snake.pop(0)

        if self.display:
            print(f"Grid:\n{self.grid}\n")
            print(f"Snake: {self.snake}\n")

        head = self.snake[-1]

        # Wall Collision
        if head[0] >= self.size or head[1] >= self.size or head[0] < 0 or head[1] < 0:
            self.crashed = True
        else:
            self.grid[head[0]][head[1]] = 1

        # Snake Collision
        if head in self.snake[:-1]:
            self.crashed = True

        return move

    def run(self, display=False):
        while True:

            if self.crashed == True:
                pygame.quit()
                return -1

            if self.iter >= self.max_iter:
                pygame.quit()
                return 0

            moves = self.move_snake()

            self.iter += 1

            if display:
                self.display_pygame()

    def display_pygame(self):
        """Display the game in a pygame window
        """
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self._coeff = SCREEN_SIZE // self.size
        self._screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self._font = pygame.font.SysFont("Monospace", 18, bold=True)
        self._clock = pygame.time.Clock()

        self._screen.fill(COLORS.get("white"))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == 1:
                    pygame.draw.rect(
                        self._screen,
                        COLORS.get("green"),
                        [
                            j * self._coeff,
                            i * self._coeff,
                            1 * self._coeff,
                            1 * self._coeff,
                        ],
                    )
                elif self.grid[i][j] == 2:
                    pygame.draw.rect(
                        self._screen,
                        COLORS.get("red"),
                        [
                            j * self._coeff,
                            i * self._coeff,
                            1 * self._coeff,
                            1 * self._coeff,
                        ],
                    )

        pygame.display.update()
        self._clock.tick(10)
