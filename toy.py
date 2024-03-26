from loss import *

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn


class LinearNN(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation='relu'):
        super(LinearNN, self).__init__()
        self.hidden = hidden_size
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, embed),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, data):
        emb = self.encoder(data)
        emb = self.dropout(emb)  # Apply dropout after encoding
        output = self.decoder(emb)
        return output

    def get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError("Invalid activation function")


# Example usage:
# Define your model
model = LinearNN(in_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, out_size=OUTPUT_SIZE, embed=30)

# Define your loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification

# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    # Forward pass
    outputs = model(inputs)

    # Compute loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
