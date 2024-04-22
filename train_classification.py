# Make sure we're using a NVIDIA GPU
from typing import Tuple
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import os
import torchvision
import wandb
import torch


# Check available GPU memory and total GPU memory
total_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
print(f"Total free GPU memory: {round(total_free_gpu_memory * 1e-9, 3)} GB")
print(f"Total GPU memory: {round(total_gpu_memory * 1e-9, 3)} GB")

GPU_SCORE = torch.cuda.get_device_capability(0)
if GPU_SCORE >= (8, 0):
    print(
        f"[INFO] Using GPU with score: {GPU_SCORE}, enabling TensorFloat32 (TF32) computing (faster on new GPUs)"
    )
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    print(
        f"[INFO] Using GPU with score: {GPU_SCORE}, TensorFloat32 (TF32) not available, to use it you need a GPU with score >= (8, 0)"
    )
    torch.backends.cuda.matmul.allow_tf32 = False


wandb.login()


device = "cuda" if torch.cuda.is_available() else "cpu"


# Set batch size depending on amount of GPU memory
total_free_gpu_memory_gb = round(total_free_gpu_memory * 1e-9, 3)
if total_free_gpu_memory_gb >= 16:
    # Note: you could experiment with higher values here if you like.
    BATCH_SIZE = 128
    IMAGE_SIZE = 224
    print(
        f"GPU memory available is {total_free_gpu_memory_gb} GB, using batch size of {BATCH_SIZE} and image size {IMAGE_SIZE}"
    )
else:
    BATCH_SIZE = 32
    IMAGE_SIZE = 128
    print(
        f"GPU memory available is {total_free_gpu_memory_gb} GB, using batch size of {BATCH_SIZE} and image size {IMAGE_SIZE}"
    )


def create_model(num_classes=10):
    """
    Creates a ResNet50 model with the latest weights and transforms via torchvision.
    """
    model_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    transforms = model_weights.transforms()
    model = torchvision.models.resnet50(weights=model_weights)

    # Adjust the number of output features in model to match the number of classes in the dataset
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    return model, transforms


def get_train_test_dataloader(transforms, batch_size: int, num_workers: int):
    train_dataset = torchvision.datasets.CIFAR10(
        root=".", train=True, download=True, transform=transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=".",
        train=False,  # want the test split
        download=True,
        transform=transforms,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader


# ## Train function


def train_step(
    epoch: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    disable_progress_bar: bool = False,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar,
    )

    for batch, (X, y) in progress_bar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1),
            }
        )
        if (batch + 1) % 20 == 0:
            wandb.log(
                {
                    "train_loss": train_loss / (batch + 1),
                    "train_accuracy": train_acc / (batch + 1),
                }
            )

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})
    return train_loss, train_acc


def test_step(
    epoch: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}",
        total=len(dataloader),
    )

    # Turn on inference context manager
    with torch.inference_mode():  # no_grad() required for PyTorch 2.0, I found some errors with `torch.inference_mode()`, please let me know if this is not the case
        # Loop through DataLoader batches
        for batch, (X, y) in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                    "test_acc": test_acc / (batch + 1),
                }
            )

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
):
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        # Perform training step and time it
        train_epoch_start_time = time.time()
        train_loss, train_acc = train_step(
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        train_epoch_end_time = time.time()
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        # Perform testing step and time it
        test_epoch_start_time = time.time()
        test_loss, test_acc = test_step(
            epoch=epoch,
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        test_epoch_end_time = time.time()
        test_epoch_time = test_epoch_end_time - test_epoch_start_time

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"train_epoch_time: {train_epoch_time:.4f} | "
            f"test_epoch_time: {test_epoch_time:.4f}"
        )


def test(model, dataloader):
    test_acc = 0
    model.eval()
    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing",
        total=len(dataloader),
    )
    with torch.inference_mode():
        for _, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
        test_acc = test_acc / len(dataloader)
        print(f"Test accuracy: {test_acc}")
        wandb.log({"test_accuracy": test_acc})
    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, f="model.onnx")
    wandb.save("model.onnx")


config = dict(
    epochs=5,
    classes=10,
    learning_rate=2e-3,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=os.cpu_count(),
    weight_decay=0.005,
)
wandb.init(project="new-sota-model", name="dat-test-resnet50-v4", config=config)


model, transforms = create_model(num_classes=config["classes"])
model.to(device)
compiled_model = torch.compile(model)
train_dataloader, test_dataloader = get_train_test_dataloader(
    transforms, config["batch_size"], config["num_workers"]
)
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=config["learning_rate"],
#     weight_decay=config["weight_decay"],
#     amsgrad=True,
# )
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
    momentum=0.65,
)
loss_fn = torch.nn.CrossEntropyLoss()
train(
    model=compiled_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=config["epochs"],
    device=device,
)
torch.save(
{"model_state_dict": model.state_dict(),
"optimizer_state_dict": optimizer.state_dict(),
},
f"compile_model_v4.pt")
