import torch
import matplotlib.pyplot as plt
import numpy as np
from LearningDeeply import SoooooooDeeeeeeep, LabelGenerator


def plot_decision_boundary(model):
    inputs = torch.zeros((10000, 2))
    for i in range(100):
        for j in range(100):
            inputs[i * 100 + j, 0] = i/10 - 5
            inputs[i * 100 + j, 1] = j/10 - 5
    Z = model(inputs)
    Z = Z.detach().numpy().reshape((100, 100))  
    Z = (Z > 0.5).astype(int)
    # Plot the contour and training examples
    plt.contourf(Z, cmap='gray')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()


model = torch.load("mymodel.pth")
plot_decision_boundary(model)