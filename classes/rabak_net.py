from torch import nn
import torch.nn.functional as F


class RabakNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.2):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.activation = F.relu
        self.dropout = nn.Dropout(p = drop_p)

    def forward(self, x):
        for fc in self.hidden_layers:
            x = self.dropout(self.activation(fc(x)))
        return F.log_softmax(self.output(x), dim = 1)