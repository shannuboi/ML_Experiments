import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def LabelGenerator(inputs):
    return (inputs[:, 0] > inputs[:, 1]**2).int().float()

def SetUpData(batch_size):
    inputs = torch.rand((1000, 2)) * 10 - 5

    labels = LabelGenerator(inputs)
    dataset = TensorDataset(inputs, labels.unsqueeze(1))
    m_train = 900
    m_test = 100
    train_set, test_set = random_split(dataset, [m_train, m_test])

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size)

    return train_loader, test_loader


def run_1_epoch(model, loss_fn, optimizer, loader, train = False):  
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct_preds = 0  

    for batch, labels in tqdm(loader):
        # Zeroing out the gradients for parameters
        if train:
            optimizer.zero_grad()
            
        # Forward pass on the input batch
        output = model(batch)

        # Acquire predicted class indices
        predicted = (output > 0.5).int().float()        
        # Compute the loss for the minibatch
        loss = loss_fn(output, labels)        
        # Backpropagation
        if train:
            loss.backward()
            optimizer.step()
        
        # Extra variables for calculating loss and accuracy
        # count total predictions for accuracy calcutuon for this epoch
        total_correct_preds += (predicted == labels).sum().item()
        total_loss += loss.item()
    

    total_batches_in_loader = len(loader)  
    loss = total_loss / total_batches_in_loader

    total_samples_in_loader = len(loader.dataset)
    accuracy = 100 * total_correct_preds / total_samples_in_loader

    return loss, accuracy 


def fit(epochs, lr, model, train_loader, test_loader, opt_fn, loss_fn):
    optimizer = opt_fn(model.parameters(), lr)
    history = {'train': [], 'test': []}

    for i in range(epochs):    
        print("Epoch %d: Train"%(i))
        train_loss, train_accuracy = run_1_epoch(model, loss_fn, optimizer, train_loader, train=True)

        print("Epoch %d: Testing"%(i))
        with torch.no_grad():
            test_loss, test_accuracy = run_1_epoch(model, loss_fn, optimizer, test_loader, train=False)

        print('Train loss: {:.2f}   Train accuracy: {:.2f}'.format(train_loss, train_accuracy))
        print('Test loss:  {:.2f}   Test accuracy:  {:.2f}'.format(test_loss, test_accuracy))

        history['train'].append((train_loss, train_accuracy)) 
        history['test'].append((test_loss, test_accuracy))
    
    return history


def Predict(model, x):
    x = torch.Tensor(list(x)).unsqueeze(0)
    output = model(x)
    print("Input:", x.detach().numpy()[0])
    print("Model predicted: ", output.detach().numpy()[0])
    print("Actual Output: ", LabelGenerator(x).detach().numpy()[0])


class SoooooooDeeeeeeep(nn.Module):
    def __init__(self, layers):
        assert(len(layers) > 1)
        super().__init__()
        self.hidden_layers = []
        for i in range(1, len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i-1], layers[i]))
            self.hidden_layers.append(nn.ReLU())
        
        self.output_linear = nn.Linear(layers[-2], layers[-1])
        self.output_activation = nn.Sigmoid()    

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_linear(x)
        x = self.output_activation(x)

        return x


if __name__ == "__main__":
    model = SoooooooDeeeeeeep([2, 32, 8, 1])
    training, testing = SetUpData(64)
    fit(100, 0.01, model, training, testing, torch.optim.Adam, binary_cross_entropy)
    torch.save(model,'mymodel.pth')   
    # model = torch.load("mymodel.pth")
    # Predict(model, (12, 17))