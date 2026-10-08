# train the CNN of my own architecture (part 1)
import json
import os
import random

os.environ["KERAS_BACKEND"] = "torch"

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

# --- CONSTANTS ---
RANDOM_SEED = 42
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def seed_everything(seed=42):
    """Fixes seeds to reproduce outcomes. Returns a NumPy random generator."""
    # fmt: off
    random.seed(seed)                                       # seeds Python's built-in "random" module
    np.random.seed(seed)                                    # legacy global stream (some libs use it)
    keras.utils.set_random_seed(seed)                       # python + numpy + torch backend
    os.environ["PYTHONHASHSEED"] = str(seed)                # seeds child processes of DataLoaders with multiple background workers (num_workers > 0)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                        # if using multi-GPU

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # gives cuBLAS (the matrix-multiplication library) a chunk of memory to store intermediate sums
    torch.backends.cudnn.benchmark = False                  # forces to use only one hard-coded CNN algorithm instead of searching for the fastest every time
    torch.backends.cudnn.deterministic = True               # makes the previous line choose the algorithm that doesn't rely on atomic addition (slows training by ~14%; scales even more for heavy nets)
    torch.use_deterministic_algorithms(True)                # forces every other single PyTorch function to use a deterministic path (scatter, gather, interpolations, pooling etc.)

    rng = np.random.default_rng(seed)                       # # isolated Generator to pass around explicitly
    return rng


def plot_loss_acc_per_ep(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """Plots the metrics for training and validation performance comparison."""
    print("Smoothed scores over epochs grouped by folds:")
    for ep, (tl, vl, ta, va) in enumerate(zip(train_losses, valid_losses, train_accuracies, valid_accuracies)):
        print(f"Epoch {ep:>2}: train_loss={tl:.4f}  val_loss={vl:.4f}  train_acc={ta:.4f}  val_acc={va:.4f}")

    print()
    _fix, axes = plt.subplots(ncols=2, figsize=(15, 4))

    # --- first plot: train and validation losses ---
    axes[0].plot(np.arange(len(train_losses)), train_losses, ".-")
    axes[0].plot(np.arange(len(valid_losses)), valid_losses, ".-")
    axes[0].legend(["train", "validation"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer
    axes[0].grid()

    # --- second plot: train and validation accuracies ---
    axes[1].plot(np.arange(len(train_accuracies)), train_accuracies, ".-")
    axes[1].plot(np.arange(len(valid_accuracies)), valid_accuracies, ".-")
    axes[1].legend(["train", "validation"])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer
    axes[1].grid()


class FashionMNISTDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = torch.from_numpy(data).clone().float()  # pixels to floats to be able to normalize later
        self.targets = torch.from_numpy(targets).clone().int()  # class labels are integers (0-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target


def cnn_block(n, activation_type="relu"):
    """Creates a sequential block consisting of Conv2D -> BatchNorm -> Activation (ResNet-like)."""
    return keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=n,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",  # padding=0
                use_bias=False,  # no use in bias before BatchNorm
                activation=None,  # no use in activation before BatchNorm
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation_type),
        ]
    )


def build_final_model(base_lr: float, steps: int, normalizer):
    inputs = keras.layers.Input(shape=(28, 28))
    x = keras.layers.Reshape((28, 28, 1))(inputs)
    x = normalizer(x)

    x = cnn_block(64, "relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = cnn_block(2 * 64, "relu")(x)
    x = cnn_block(4 * 64, "relu")(x)
    x = cnn_block(6 * 64, "relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    lr = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[steps * 10, steps * 15], values=[base_lr, base_lr * 0.1, base_lr * 0.01])
    outputs = keras.layers.Dense(units=10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # --- SEEDING FUNCTIONALITY ---
    _rng = seed_everything(RANDOM_SEED)

    torch_gen = torch.Generator()
    torch_gen.manual_seed(RANDOM_SEED)

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train_writable = X_train.copy()
    y_train_writable = y_train.copy()

    X_test_writable = X_test.copy()
    y_test_writable = y_test.copy()

    full_train_ds = FashionMNISTDataset(X_train_writable, y_train_writable)
    test_ds = FashionMNISTDataset(X_test_writable, y_test_writable)

    # Split train into train/validation (90/10)
    train_size = int(0.9 * len(X_train))
    val_size = len(full_train_ds) - train_size

    train_ds, _valid_ds = random_split(
        dataset=full_train_ds,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, generator=torch_gen)

    # Calculate the whole dataset's mean and std in batches (not needed to do it in batches for such a small dataset though)
    # (z-score normalization)
    n = 0
    s = 0.0
    ss = 0.0
    for x, _ in train_loader:
        s += x.sum().item()
        ss += (x**2).sum().item()
        n += x.numel()

    mean_ds = s / n
    std_ds = (ss / n - mean_ds**2) ** 0.5
    variance_ds = std_ds**2

    # Hardcode the stats into Keras
    normalizer = keras.layers.Normalization(
        axis=None,
        mean=mean_ds,
        variance=variance_ds,
    )

    # Fit on full train before assessing on test:
    keras.utils.set_random_seed(RANDOM_SEED)
    torch_gen.manual_seed(RANDOM_SEED)

    full_train_loader = DataLoader(full_train_ds, batch_size=125, shuffle=True, generator=torch_gen)
    steps = len(full_train_loader)  # batches per epoch (for lr scheduler)
    model = build_final_model(base_lr=0.0006, steps=steps, normalizer=normalizer)

    history = model.fit(
        x=full_train_loader,
        epochs=16,
        verbose=1,
        shuffle=False,
    )

    plot_loss_acc_per_ep(history.history["loss"], [], history.history["accuracy"], [])

    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    test_loss, test_acc = model.evaluate(test_loader, verbose=1)

    print("\n" + "=" * 40)
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"FINAL TEST LOSS:     {test_loss:.4f}")
    print("=" * 40)

    # --- Save (Keras 3 native: architecture + weights + optimizer state in one file) ---
    model.save("task1_final_full_train.keras")

    # --- Save the training history + test scores (for the Dash app to plot) ---
    with open("task1_final_full_train_history.json", "w") as f:
        json.dump(
            {**{k: [float(v) for v in vals] for k, vals in history.history.items()}, "test_loss": float(test_loss), "test_accuracy": float(test_acc)},
            f,
            indent=2,
        )

    # --- Load ---
    # loaded = keras.models.load_model("task1_final_full_train.keras")

    # --- 1. Predictions + true labels out of the DataLoader ---
    y_prob = model.predict(test_loader, verbose=0)  # (10000, 10) softmax probabilities
    y_pred = y_prob.argmax(axis=1)  # most probable class indices
    y_true = np.concatenate([yb.numpy() for _, yb in test_loader])  # convert each y_batch from tensor to numpy and concatenate into 1D array

    # --- 2. Per-class metrics ---
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # --- 3. Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)

    _fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",  # integer format
        xticks_rotation=45,
    )
    ax.set_title("Fashion-MNIST — Raw Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
