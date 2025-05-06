import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import DataFrame
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# preprocessing the data

def preprocessing_data(csv_path: str,remove_id:bool = True) -> pd.DataFrame:
    """
    Preprocesses the titanic CSV file. Dropping irrelevant columns and filling in missing values.
    :param csv_path: path to the csv file
    :return: Panda DataFrame
    """
    df = pd.read_csv(csv_path)
    df["Sex"] = df.Sex.map(lambda x: 1 if x == "female" else 0)  # encoding the gender
    df["Embarked"] = df.Embarked.map({"C": 0, "Q": 1, "S": 2})  # Encoding the embark port
    df = df.drop(["Name", "Cabin", "Ticket"], axis=1) #Dropping the names, cabins and Ticket
    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(2)  # Most common value
    return df

def generate_batch_tensor(df: pd.DataFrame) -> (torch.Tensor, torch.Tensor | None):
    """
    Converts and splits the dataframe into input tensor and target tensor.
    :param df: dataframe to be converted.
    :return: A tuple consisting of the input tensor (float32) and the target tensor (long). if no Survived coloumn is
    provided, return input tensor and None.
    """
    if "Survived" not in df.columns:
        return torch.tensor(df.values.astype(np.float32),dtype=torch.float32), None
    input_set = df.drop("Survived", axis=1).values.astype(np.float32)
    target_set = df["Survived"].values.astype(np.long)


    input_tensor = torch.tensor(input_set, dtype=torch.float32)
    target_tensor = torch.tensor(target_set, dtype=torch.long)
    return input_tensor, target_tensor

titanic_train_data = preprocessing_data("train.csv")
input_tensor, target_tensor = generate_batch_tensor(titanic_train_data)
dataset = TensorDataset(input_tensor, target_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

OUTPUT_SIZE = 2 #Either survived or not
INPUT_SIZE = input_tensor.shape[1]

LEARNING_RATE = 0.001

class TitanicGuesser(nn.Module):
    """
    Neural network to predict the survival status of Titanic passengers.
    The experiments showed a relatively complex (more hidden layers) network helps the convergence in this project
    """
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
        )

    def forward(self,x):
        return self.net(x)

model = TitanicGuesser(INPUT_SIZE,64,OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
losses = [] # Tracking the loss of the model
acc = [] # Tracking the accuracy of the model


for epoch in range(1200):
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
    print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
    writer.add_scalar("avg_loss",avg_loss,epoch)

print("Training complete and plots generated.")
writer.close()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Titanic Survivorship Training Phase")
plt.savefig("loss.png")
plt.figure()
plt.plot(acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Titanic Survivorship Training Phase")
plt.savefig("acc.png")

# inference phase
test_path = "test.csv"
passenger_ids = pd.read_csv(test_path)["PassengerId"]
#preparing data
df = preprocessing_data(test_path)

input_tensor, _ = generate_batch_tensor(df)
# model.eval() #evaluation mode
with torch.no_grad():
    output = model(input_tensor)
output_probs = torch.softmax(output,dim=1)
# predictions = (output_probs > 0.5).float()
class_labels = torch.argmax(output_probs, dim=1).tolist()

result_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    'Survived':    class_labels
})
df = result_df
df.to_csv("survived.csv", index=False)
print("Surivorship infered and saved into survived.csv. Done!")


