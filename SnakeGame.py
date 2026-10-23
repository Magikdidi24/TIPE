import numpy as np

SCREEN_SIZE = 600
COLORS = {"white": (255, 255, 255), "green": (0, 255, 0), "red": (255, 0, 0)}

# Déplacements (ligne, colonne) : la grille est indexée grid[ligne][colonne]
LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)

MOVES = [LEFT, RIGHT, UP, DOWN]

# Contenu des cases de la grille
EMPTY, SNAKE, APPLE = 0, 1, 2

SNAKE_SIZE = 3  # longueur initiale du serpent (comme dans la version C)


class RandomPlayer:
    """Joueur de référence : un déplacement au hasard parmi les 3 permis (pas de demi-tour)."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def get_action(self, snake, grid):
        (y_neck, x_neck), (y_head, x_head) = snake[-2], snake[-1]
        back = (y_neck - y_head, x_neck - x_head)
        moves = [move for move in MOVES if move != back]
        return moves[self.rng.integers(len(moves))]


class SnakeGame:
    """Une partie de Snake sur une grille size x size.

    - le serpent est une liste de cases (ligne, colonne), la tête est snake[-1] ;
    - une seule pomme à la fois, placée au hasard sur une case libre ;
    - la partie s'arrête si le serpent touche un mur ou son corps, s'il reste
      max_iter déplacements sans manger (famine : évite les serpents qui tournent
      en rond indéfiniment) ou si la grille est pleine.

    Le joueur (player) doit fournir get_action(snake, grid) -> un élément de MOVES.
    """

    def __init__(self, size: int, snake_size: int, player, max_iter=100, seed=None):
        if size // 2 + snake_size > size:  # le serpent de départ va du centre vers la droite
            raise ValueError("grille trop petite pour le serpent de départ")
        self.size = size
        self.snake_size = snake_size
        self.player = player
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.iter = 0  # nombre total de déplacements
        self.iter_without_apple = 0  # déplacements depuis la dernière pomme
        self.score = 0  # nombre de pommes mangées
        self.crashed = False

        self.grid = np.zeros([self.size, self.size], dtype=int)

        self.snake = [
            (self.size // 2, self.size // 2 + i) for i in range(self.snake_size)
        ]
        for coord in self.snake:
            self.grid[coord] = SNAKE
        self.direction = RIGHT  # la tête est à droite du corps

        self.apple = self.generate_apple()
        self.grid[self.apple] = APPLE

    def generate_apple(self):
        """Renvoie une case libre tirée au hasard (None si la grille est pleine)."""
        free = np.argwhere(self.grid == EMPTY)
        if len(free) == 0:
            return None
        i, j = free[self.rng.integers(len(free))]
        return (int(i), int(j))

    def move_snake(self):
        move = self.player.get_action(self.snake, self.grid)

        # Comme dans la version C : un demi-tour est ignoré, on continue tout droit
        if move == (-self.direction[0], -self.direction[1]):
            move = self.direction
        self.direction = move

        head = (self.snake[-1][0] + move[0], self.snake[-1][1] + move[1])
        self.iter += 1
        self.iter_without_apple += 1

        # Wall collision
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size):
            self.crashed = True
            return move

        eating = head == self.apple
        if eating:
            self.score += 1
            self.iter_without_apple = 0
        else:
            # Delete the tail (avant le test de collision : la case libérée est libre)
            tail = self.snake.pop(0)
            self.grid[tail] = EMPTY

        # Snake collision
        if self.grid[head] == SNAKE:
            self.crashed = True
            return move

        self.snake.append(head)
        self.grid[head] = SNAKE

        if eating:
            self.apple = self.generate_apple()
            if self.apple is not None:
                self.grid[self.apple] = APPLE

        return move

    def is_over(self):
        return (
            self.crashed
            or self.iter_without_apple >= self.max_iter
            or self.apple is None
        )

    def run(self, display=False, fps=10):
        """Joue la partie jusqu'au bout et renvoie le score (pommes mangées)."""
        while not self.is_over():
            self.move_snake()
            if display:
                self.display_pygame(fps)
        if display:
            import pygame

            pygame.quit()
        return self.score

    def display_pygame(self, fps=10):
        """Display the game in a pygame window"""
        import pygame  # import local : l'entraînement (sans affichage) n'a pas besoin de pygame

        if not hasattr(self, "_screen"):
            pygame.init()
            self._coeff = SCREEN_SIZE // self.size
            self._screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        pygame.display.set_caption(f"Snake - score : {self.score}")
        self._screen.fill(COLORS.get("white"))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == EMPTY:
                    continue
                color = COLORS.get("green") if self.grid[i][j] == SNAKE else COLORS.get("red")
                pygame.draw.rect(
                    self._screen,
                    color,
                    [j * self._coeff, i * self._coeff, self._coeff, self._coeff],
                )

        pygame.display.update()
        self._clock.tick(fps)
