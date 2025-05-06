import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# preprocessing the data

titanic_cvf = pd.read_csv("train.csv")

titanic_cvf["Sex"] = titanic_cvf.Sex.map(lambda x: 1 if x == "female" else 0) #encoding the gender
titanic_cvf["Embarked"] = titanic_cvf.Embarked.map({"C": 0, "Q": 1, "S": 2}) #Encoding the embark port
titanic_cvf = titanic_cvf.drop(["PassengerId","Name","Cabin","Ticket"], axis=1) #Dropping the IDs, names, cabins and Ticket
# Handle missing values
titanic_cvf["Age"].fillna(titanic_cvf["Age"].median(), inplace=True)
titanic_cvf["Embarked"].fillna(2, inplace=True)  # Most common value

test_set = titanic_cvf["Survived"].values.astype(np.int64)
training_set =titanic_cvf.drop("Survived",axis=1).values.astype(np.float32)


training_tensor = torch.tensor(training_set,dtype=torch.float32)
test_tensor = torch.tensor(test_set,dtype=torch.long)

dataset = TensorDataset(training_tensor, test_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

OUTPUT_SIZE = 2 #Either survived or not
INPUT_SIZE = training_set.shape[1]

LEARNING_RATE = 0.001

class TitanicGuesser(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            # nn.Linear(input_size,hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size,output_size),
        )

    def forward(self,x):
        return self.net(x)

model = TitanicGuesser(INPUT_SIZE,64,OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
losses = []
acc = []
for epoch in range(1000):
    total_loss = 0
    accuracy_total = 0
    correct_predictions = 0
    total_samples = 0
    for batch_index, (input_vector,target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(input_vector)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_vector.size(0)
        # Compute accuracy
        predicted = torch.argmax(output, dim=1)
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss / len(dataset)
    losses.append(avg_loss)
    epoch_accuracy = correct_predictions / total_samples
    acc.append(epoch_accuracy)
    print(f"Epoch {epoch:03d} — avg loss: {avg_loss:.4f}")
    writer.add_scalar("avg_loss",avg_loss,epoch)


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Titanic Survivorship Training Phase")
plt.savefig("loss.png")

plt.plot(acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Titanic Survivorship Training Phase")
plt.savefig("loss.png")
