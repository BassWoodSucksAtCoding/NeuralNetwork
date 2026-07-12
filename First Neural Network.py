import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Setting up hyperparameters here so that its easy to change
learning_rate = 1e-3
batch_size = 32
epochs = 10

# Setting the training data from an existing dataset called MNIST containing digits
training_data = datasets.MNIST(
    root="data",
    train=True, # Enabling train mode
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # Turning images to numbers more easily proccessable by the nn
)

# Setting the testing data which are separate from the training data so we can see how accurate
# the model is
testing_data = datasets.MNIST(
    root="data",
    train=False, # Enabling test mode
    download=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

# Making a dataloader for each using the respective data to easily load the data
training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# The model class itself
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # Getting the init of the parent
        self.flatten = nn.Flatten() # Flattening the tensor into 1d features

        # The steps the nn will follow to learn
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Sigmoid(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    # What to do upon getting an input
    def forward(self, x):
        x = self.flatten(x) # Flattening
        logits = self.linear_relu_stack(x) # Going through the steps
        return logits # Returning the logits

# This function will be called when training
def train_loop(model, optimiser, loss_fn, dataloader):

    # Setting the model to train (unnecessary here) and finding the size of the model
    size = len(dataloader.dataset)
    model.train()

    # X are in the inputs or images, y are the labels or correct answers, batch is the natch no
    for batch,(X,y) in enumerate(dataloader):
        pred = model(X) # Getting predicted result for each image
        loss = loss_fn(pred, y) # CrossEntropy automatically turns logits into probabilities so no need for Softmax

        # Backpropagation to calculate gradients and learn
        loss.backward() # Go through each operation
        optimiser.step() # Learn
        optimiser.zero_grad() # Remove gradients

        # Update how much is done every 100 images
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# This function will be called when testing
def test_loop(dataloader, model, loss_fn):
    # eval sets the model to test mode (unnecessary here)
    model.eval()
    size = len(dataloader.dataset) # Size of dataset
    num_batches = len(dataloader) # Number of batches
    test_loss, correct = 0, 0 # How many correct and the sum of loss

    # Turning off grad calculation so that the model does not learn
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # Finding if the prediction is correct, converting into floating numbers, finding 
            # the sum, and converting the tensor into a number

    # Output
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Actually running everything
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Going through 10 epochs
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_loop(model,optimiser,loss_fn,training_dataloader)
    test_loop(testing_dataloader, model, loss_fn)

print("Done!")

