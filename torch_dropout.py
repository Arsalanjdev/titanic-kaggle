import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from preprocess import get_sets
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1000
HIDDEN_SIZE = 64


class TitanicModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        return self.net(x)


def train_model(feature_tensor: torch.Tensor, target_tensor: torch.Tensor):
    # Load and prepare data
    # input_tensor, target_tensor = get_sets("train.csv")
    dataset = TensorDataset(feature_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer and loss
    model = TitanicModel(
        feature_tensor.shape[1], HIDDEN_SIZE, 1
    )  # Output size 1 for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()
    losses: list[int] = []
    accuracies: list[int] = []
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (preds.squeeze() == targets).sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(dataset)
        accuracy = correct_predictions / len(dataset)

        # Logging
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        losses.append(avg_loss)
        accuracies.append(accuracy)
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}"
            )

    writer.close()

    # Save Loss Plot
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"training_loss.png")  # Save to file
    plt.close()  # Close the figure to free memory

    # Save Accuracy Plot
    plt.figure()
    plt.plot(accuracies, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"training_accuracy.png")  # Save to file
    plt.close()

    return model


def evaluate_model(model, test_path="test.csv", preprocessor=None):
    # Load test data
    passenger_ids = pd.read_csv(test_path)["PassengerId"]
    features, targets, _ = get_sets(test_path, is_test=True)

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze()

    # Save results
    result_df = pd.DataFrame(
        {"PassengerId": passenger_ids, "Survived": predictions.cpu().numpy()}
    )
    result_df.to_csv("torch_dropout_predictions.csv", index=False)

    # Calculate metrics if targets are available
    if targets is not None:
        print(
            classification_report(
                targets.cpu().numpy(),
                predictions.cpu().numpy(),
                target_names=["Did Not Survive", "Survived"],
            )
        )

    return result_df


if __name__ == "__main__":
    # Train and evaluate
    X_train, y_train, preprocessor = get_sets("train.csv")
    trained_model = train_model(X_train, y_train)
    predictions = evaluate_model(trained_model, preprocessor=preprocessor)
    print("Training complete. Predictions saved to torch_dropout_predictions.csv")
