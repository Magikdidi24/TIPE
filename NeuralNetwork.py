import numpy as np

from SnakeGame import APPLE, MOVES

# Ce que voit le serpent pour chaque contenu de case : vide -> 0, serpent -> -1, pomme -> 1
VISION = np.array([0, -1, 1])


def generate_indiv(input_size: int, hidden_size: int, output_size: int, rng):
    """Un individu (son génome) = les poids du réseau : 3 matrices tirées dans [-1, 1].

    Couche cachée 1, couche cachée 2, couche de sortie. La dernière colonne de
    chaque matrice est le biais (il multiplie une entrée constante égale à 1).
    """
    return [
        rng.uniform(-1, 1, (hidden_size, input_size + 1)),
        rng.uniform(-1, 1, (hidden_size, hidden_size + 1)),
        rng.uniform(-1, 1, (output_size, hidden_size + 1)),
    ]


def forward(indiv, input_vector):
    """Propagation avant : sortie d'un neurone = tanh(produit scalaire poids . entrées + biais)."""
    x = input_vector
    for layer in indiv:
        x = np.tanh(layer[:, :-1] @ x + layer[:, -1])
    return x


class NeuralPlayer:
    """Joueur de Snake piloté par un réseau de neurones (perceptron multicouche)."""

    def __init__(self, indiv, window_size: int):
        self.indiv = indiv
        self.window_size = window_size

    def visualize_grid(self, grid: np.ndarray, y_head: int, x_head: int):
        """Fenêtre window_size x window_size centrée sur la tête, aplatie en vecteur.

        Les murs (hors de la grille) sont vus comme le corps du serpent : -1.
        """
        r = self.window_size // 2
        vision = np.full((self.window_size, self.window_size), -1)
        # partie de la fenêtre qui est dans la grille : lignes i0..i1, colonnes j0..j1
        i0, i1 = max(y_head - r, 0), min(y_head + r + 1, grid.shape[0])
        j0, j1 = max(x_head - r, 0), min(x_head + r + 1, grid.shape[1])
        vision[i0 - y_head + r : i1 - y_head + r, j0 - x_head + r : j1 - x_head + r] = VISION[
            grid[i0:i1, j0:j1]
        ]
        return vision.flatten()

    def apple_direction(self, grid: np.ndarray, y_head: int, x_head: int):
        """4 entrées, dans l'ordre de MOVES : 1 si la pomme est à gauche / à droite / en haut / en bas."""
        # indice de la (seule) case pomme dans la grille aplatie -> (ligne, colonne)
        y_apple, x_apple = divmod(int((grid == APPLE).argmax()), grid.shape[1])
        return np.array(
            [x_apple < x_head, x_apple > x_head, y_apple < y_head, y_apple > y_head], dtype=float
        )

    def get_action(self, snake, grid: np.ndarray):
        y_head, x_head = snake[-1]
        input_vector = np.concatenate(
            [self.visualize_grid(grid, y_head, x_head), self.apple_direction(grid, y_head, x_head)]
        )
        output = forward(self.indiv, input_vector)
        return MOVES[np.argmax(output)]
