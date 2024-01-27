import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0", ]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
import torchvision.transforms.v2 as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import wandb


# Your code here...


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    batch_size = 28
    if kind == 'train':
        aug = A.Compose([
            A.RandomHorizontalFlip(p=0.5),
            A.RandomVerticalFlip(p=0.1),
            A.RandomGrayscale(p=0.1),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(root='./tiny-imagenet-200/train', transform=aug)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        return loader
    else:
        aug = A.Compose([
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if kind == 'val':
            dataset = ImageFolder(root='./tiny-imagenet-200/val', transform=aug)
        if kind == 'test':
            dataset = ImageFolder(root='./tiny-imagenet-200/test', transform=aug)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )
        return loader


def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    n_classes = 200
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    return model


def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())
    return optimizer


def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        predictions = model(batch).to(device)

    model.train()
    return predictions


def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    model.eval()  # Переводим модель в режим оценки
    total_samples = 0
    correct_samples = 0
    total_loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_samples += labels.size(0)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct_samples += predicted.eq(labels).sum().item()

    accuracy = correct_samples / total_samples
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss


def save_check_point(model, filename):
    with open(filename, "wb") as fp:
        torch.save(model.state_dict(), fp)


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_crossentr = torch.nn.CrossEntropyLoss()

    wandb.login()

    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project4",
        name="adam + res34 - atempt2",
        # track hyperparameters and run metadata
        reinit=True,
        config={
            "learning_rate": 0.001,
            "architecture": "CNN",
            "dataset": "Small_ImageNet",
            "epochs": 10,
        }
    )

    total_loss = 0
    total_acc = 0
    total_n = 0
    model.train()
    model.to(device)

    for epoch in range(21):
        total_loss = 0
        total_loss_val = 0
        model.train()
        for batch_data, batch_labels in train_dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            model_labels = model(batch_data)
            # model_prediction = model.predict(batch_data)
            new_loss = torch_crossentr(model_labels, batch_labels)

            optimizer.zero_grad()
            new_loss.backward()
            optimizer.step()
            one_batch_loss = float(torch_crossentr(model_labels, batch_labels))
            #print(one_batch_loss)
            wandb.log({"loss": one_batch_loss})

            total_loss += one_batch_loss

            wandb.log({"epoch_loss": total_loss})

        save_check_point(model, "./checkpoint.pth")
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        #save_check_point(model, "./checkpoint.pth")

        with torch.no_grad():

            for batch_data, batch_labels in val_dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                model_labels = model(batch_data)
                # model_prediction = model.predict(batch_data)
                new_loss = torch_crossentr(model_labels, batch_labels)
                wandb.log({"loss_val": new_loss})
                total_loss_val += new_loss

            wandb.log({"epoch_loss": total_loss_val})


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    with open(checkpoint_path, "rb") as fp:
        state_dict = torch.load(fp)
    model.load_state_dict(state_dict)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"

    md5_checksum = "f2d21161e1d276f11799c1c790df9626"
    google_drive_link = "https://drive.google.com/drive/folders/1zTl_Vkce7T8u-ATCdGGvcbTXEWgImoNR"
    return md5_checksum, google_drive_link
