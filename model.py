import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Se crea la red neural extendiendo de torch.nn,module


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    """
    Al heredar se hace necesario la implemetanción de un metodo llamado 
    forward, donde se construye la arquitectura de la red y será el 
    encargado de predecir
    """

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    """
    función para ir guardando el modelo cada que se encargue un max_score
    """

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


"""
Clase encargada de entrenar el modelo, según lo establecido por
el método Qlearning
"""


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()

    """
    La función responde a entrenar el modelo solo con un movimientos o 
    con el registro historico de movimientos
    """

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # Q
        pred = self.model(state)

        target = pred.clone()
        """
        Bucle encargado de calcular Q_new con la 
        ecuación de Bellman Q_new = r + y * mac(next Q value).

        El objetivo es encontrar el movimiento que tiene la máxima recompensa 
        en el siguiente estado.
        """
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * \
                    torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        """
        Calcula el loss comparando los valores objetivo con las predicciones del modelo
        """
        loss = self.criterion(target, pred)

        """
        Se actualizan los pesos del modelo según el loss
        """
        loss.backward()
        self.optimizer.step()
