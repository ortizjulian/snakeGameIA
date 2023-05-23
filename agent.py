import torch
import random
import os
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)
        # Se instacion el modelo
        self.model = Linear_QNet(11, 400, 3)
        # En caso de que se quiera empezar con lo que ya aprendió
        if os.path.exists('model/model.pth'):
            self.model.load_state_dict(torch.load('model/model.pth'))
            self.model.eval()
            self.init_epsilon_value = 5
        else:
            self.init_epsilon_value = 65

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Leer todas las variables que constituyen el estado que se le envia al modelo
    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Peligro derecho
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Peligro derecha
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Peligro izquierda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # En que posición va
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # En donde está la comida
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        return np.array(state, dtype=int)
    """
    Guardar lo jugado para construir el modelo según la experiencia
    """

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """
    Entrenar con registro de partidas
    """

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    """
    Entrenar con un movimiento
    """

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    """
    Predecir la acción según el estado o de vez en cuando hacer algo al azar
    a medida que avanza el juego hace menos cosas al hacer y le hace caso a la
    predicción
    """

    def get_action(self, state):
        self.epsilon = self.init_epsilon_value - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


"""
Función encargada de controlar todo,
Empezar el juego,
Entrenar,
LLevar juego a juego la serpiente.
"""


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Leer el estado en el que se encuentra
        state_old = agent.get_state(game)

        # Predecir el movimiento según el estado
        final_move = agent.get_action(state_old)

        # Mover la serpiente y leer el estado al moverse
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        """
        Se entrena por un movimiento según el estado en el que estaba,
        lo que hizo,
        la recompensa,
        al estado que llego despues de moverse,
        Y si quedo vivo o no
        """
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Entrenar entre juegos, luego de morir con todo el registro de movimientos(Experiencia)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

                # Guardar el modelo cada que se logre un mejor resultado
                agent.model.save()
            print("Game", agent.n_games, 'Max Score', record)

            # Graficar el punto nuevo
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
