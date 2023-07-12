# %%
"""
Application of ConvNext, Swin Transformer, and VGG19 for AI Generated Image Detection

Author: James Michael Fritz 

Date Created: 25/03/2023
Date Revised: 11/07/2023

This notebook demonstrates the implementation of the ConvNext, Swin Transformer,
and VGG19 to detect AI Generated Images
"""

# %%
# Imports here
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
from collections import OrderedDict
import seaborn as sns
import json
from sklearn.metrics import classification_report

# %%
# hyperparameters
BATCH_SIZE = 512
EPOCHS = 5
LR = 5e-4
LR_DECAY = 0.7
WEIGHT_DECAY = 1e-2

# %%
train_dir = "train"
test_dir = "test"

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define your transforms for the training, validation, and testing sets
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Load the dataset using ImageFolder
image_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

# Split the dataset into train and validation sets
train_size = int(0.8 * len(image_dataset))
val_size = len(image_dataset) - train_size

train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

# Create data loaders for train, validation, and test sets
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
from torch.nn import DataParallel


def fit_model(
    model, epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = DataParallel(model)

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")

        model.train()
        train_loss = []
        train_acc = []
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc.append(acc.item())

            if i % 100 == 0 and i != 0:
                print(f"Batch: {i:03d}/{len(train_dataloader)}", end=" | ")
                print(f"Train loss: {loss.item():.4f}", end=" | ")
                print(f"Train accuracy: {acc.item():.4f}", end=" | ")
                print()

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_acc) / len(train_acc)

        epoch_train_loss.append(avg_train_loss)
        epoch_train_acc.append(avg_train_acc)

        print(f"Train loss: \t\t{avg_train_loss:.4f}", end=" | ")
        print(f"Train accuracy: \t{avg_train_acc:.4f}", end=" | ")
        print()

        scheduler.step()  # update lr on every batch

        model.eval()
        val_loss = []
        val_acc = []
        for i, (inputs, labels) in enumerate(val_dataloader):
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                val_acc.append(acc.item())

        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_acc = sum(val_acc) / len(val_acc)

        epoch_val_loss.append(avg_val_loss)
        epoch_val_acc.append(avg_val_acc)

        print(f"Validation loss: \t{avg_val_loss:.4f}", end=" | ")
        print(f"Validation accuracy: \t{avg_val_acc:.4f}", end=" | ")
        print()

    history = {
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc,
    }

    return history


# %%
# Using vgg19 as base model
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# we don't want to disrupt the learned weights of the pretrained model, so we set 'requires_grad' to False
for param in vgg19.parameters():
    param.requires_grad = False

vgg19_classifier = nn.Sequential(
    OrderedDict(
        [
            ("inputs", nn.Linear(25088, 512)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(0.2)),
            ("hidden_layer1", nn.Linear(512, 128)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(0.2)),
            ("hidden_layer2", nn.Linear(128, 128)),
            ("relu3", nn.ReLU()),
            ("dropout3", nn.Dropout(0.2)),
            ("hidden_layer3", nn.Linear(128, 2)),
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)

vgg19.classifier = vgg19_classifier

optimizer_vgg19 = optim.AdamW(
    vgg19.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler_vgg19 = lr_scheduler.ExponentialLR(
    optimizer_vgg19, gamma=LR_DECAY, last_epoch=-1
)

# %%
# Using convnext as base model
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

for param in convnext.parameters():
    param.requires_grad = False

convnext_classifier = nn.Sequential(
    OrderedDict(
        [
            ("flatten", nn.Flatten()),
            ("inputs", nn.Linear(1024, 512)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(0.2)),
            ("hidden_layer1", nn.Linear(512, 128)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(0.2)),
            ("hidden_layer2", nn.Linear(128, 128)),
            ("relu3", nn.ReLU()),
            ("dropout3", nn.Dropout(0.2)),
            ("hidden_layer3", nn.Linear(128, 2)),
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)

convnext.classifier = convnext_classifier

optimizer_convnext = optim.AdamW(
    convnext.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler_convnext = lr_scheduler.ExponentialLR(
    optimizer_convnext, gamma=LR_DECAY, last_epoch=-1
)

# %%
# Using swin as base model
swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)

for param in swin.parameters():
    param.requires_grad = False

swin_classifier = nn.Sequential(
    OrderedDict(
        [
            ("flatten", nn.Flatten()),
            ("inputs", nn.Linear(1024, 512)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(0.2)),
            ("hidden_layer1", nn.Linear(512, 128)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(0.2)),
            ("hidden_layer2", nn.Linear(128, 128)),
            ("relu3", nn.ReLU()),
            ("dropout3", nn.Dropout(0.2)),
            ("hidden_layer3", nn.Linear(128, 2)),
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)

swin.head = swin_classifier

optimizer_swin = optim.AdamW(swin.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_swin = lr_scheduler.ExponentialLR(
    optimizer_swin, gamma=LR_DECAY, last_epoch=-1
)


# %%
def eval_model(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = DataParallel(model)
    model.eval()

    test_loss = []
    test_acc = []
    y_true = []
    y_pred = []

    for i, (inputs, labels) in enumerate(test_dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_true.extend(labels.cpu().numpy())

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())

            _, predictions = torch.max(outputs.data, 1)
            y_pred.extend(predictions.cpu().numpy())
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc.append(acc.item())

    avg_test_loss = sum(test_loss) / len(test_loss)
    avg_test_acc = sum(test_acc) / len(test_acc)

    print("Evaluation Results\n")
    print(f"Test loss: {avg_test_loss:.4f}", end=" | ")
    print(f"Test accuracy: {avg_test_acc:.4f}", end=" | ")
    print("\n")

    # Create a list of class names in the same order as their integer labels
    class_to_idx = image_dataset.class_to_idx
    classes = [k for k, v in class_to_idx.items()]

    # Convert integer labels to their corresponding class names
    y_true = [classes[label] for label in y_true]
    y_pred = [classes[label] for label in y_pred]

    # Print the classification report with categorical labels
    print(classification_report(y_true, y_pred, digits=4))


# %%
criterion = nn.NLLLoss()

# %%
vgg19_history = fit_model(
    model=vgg19,
    epochs=EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer_vgg19,
    scheduler=scheduler_vgg19,
)

# %%
convnext_history = fit_model(
    model=convnext,
    epochs=EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer_convnext,
    scheduler=scheduler_convnext,
)

# %%
swin_history = fit_model(
    model=swin,
    epochs=EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer_swin,
    scheduler=scheduler_swin,
)

# %%
eval_model(model=vgg19, test_dataloader=test_dataloader)

# %%
eval_model(model=convnext, test_dataloader=test_dataloader)

# %%
eval_model(model=swin, test_dataloader=test_dataloader)

# %%
sns.set_context("paper")

sns.lineplot(x=range(1, 6), y=vgg19_history["val_loss"], label="VGG19")
sns.lineplot(x=range(1, 6), y=convnext_history["val_loss"], label="ConvNext")
sns.lineplot(x=range(1, 6), y=swin_history["val_loss"], label="Swin")

plt.xticks(range(1, 6, 1))
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")

plt.legend()

# %%
sns.lineplot(x=range(1, 6), y=vgg19_history["val_acc"], label="VGG19")
sns.lineplot(x=range(1, 6), y=convnext_history["val_acc"], label="ConvNext")
sns.lineplot(x=range(1, 6), y=swin_history["val_acc"], label="Swin")

plt.xticks(range(1, 6, 1))
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Curve")

plt.legend()

# %%
torch.save(vgg19, "VGG19-CIFAKE.pth")
torch.save(convnext, "ConvNext-CIFAKE.pth")
torch.save(swin, "Swin-CIFAKE.pth")
