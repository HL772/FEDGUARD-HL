from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class FedPerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)
        self.private_head = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def create_model() -> nn.Module:
    return FedPerModel()


def load_state_dict_from_list(model: nn.Module, state_list: Dict[str, list]) -> None:
    tensor_state = {k: torch.tensor(v, dtype=torch.float32) for k, v in state_list.items()}
    model.load_state_dict(tensor_state)


def state_dict_to_list(model: nn.Module) -> Dict[str, list]:
    return {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}


def update_state_dict_from_list(model: nn.Module, state_list: Dict[str, list]) -> None:
    current_state = model.state_dict()
    for key, value in state_list.items():
        if key in current_state:
            current_state[key] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(current_state)


def extract_state_by_keys(model: nn.Module, keys: set[str]) -> Dict[str, list]:
    extracted: Dict[str, list] = {}
    for key, value in model.state_dict().items():
        if key in keys:
            extracted[key] = value.detach().cpu().tolist()
    return extracted


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    lr: float,
    device: torch.device,
    mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    label_flip: bool = False,
    num_classes: int = 10,
) -> Tuple[float, int, int, float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct = 0
    steps = 0
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if label_flip:
            batch_y = (num_classes - 1) - batch_y
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if mu > 0 and global_params is not None:
            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_params):
                prox_term += torch.sum((param - global_param) ** 2)
            loss = loss + 0.5 * mu * prox_term
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_x.size(0)
        total_samples += int(batch_x.size(0))
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == batch_y).sum().item())
        steps += 1

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, total_samples, steps, accuracy


def train_one_epoch_dual(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    lr: float,
    device: torch.device,
    mu: float = 0.0,
    global_params: Optional[Dict[str, torch.Tensor]] = None,
    private_lr: Optional[float] = None,
    private_epochs: int = 1,
    label_flip: bool = False,
    num_classes: int = 10,
) -> Tuple[float, int, int, float]:
    model.to(device)
    model.train()
    shared_params = list(model.backbone.parameters()) + list(model.head.parameters())
    optimizer_shared = torch.optim.SGD(shared_params, lr=lr)
    optimizer_private = torch.optim.SGD(model.private_head.parameters(), lr=private_lr or lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct = 0
    steps = 0
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if label_flip:
            batch_y = (num_classes - 1) - batch_y

        features = model.backbone(batch_x)
        shared_logits = model.head(features)
        shared_loss = criterion(shared_logits, batch_y)
        if mu > 0 and global_params:
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in global_params:
                    prox_term += torch.sum((param - global_params[name]) ** 2)
            shared_loss = shared_loss + 0.5 * mu * prox_term

        optimizer_shared.zero_grad()
        shared_loss.backward()
        optimizer_shared.step()

        private_logits = model.private_head(features.detach())
        private_loss = criterion(private_logits, batch_y)
        optimizer_private.zero_grad()
        private_loss.backward()
        optimizer_private.step()

        total_loss += float(private_loss.item()) * batch_x.size(0)
        total_samples += int(batch_x.size(0))
        preds = torch.argmax(private_logits, dim=1)
        correct += int((preds == batch_y).sum().item())
        steps += 1

    extra_epochs = max(int(private_epochs) - 1, 0)
    for _ in range(extra_epochs):
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if label_flip:
                batch_y = (num_classes - 1) - batch_y
            with torch.no_grad():
                features = model.backbone(batch_x)
            private_logits = model.private_head(features)
            private_loss = criterion(private_logits, batch_y)
            optimizer_private.zero_grad()
            private_loss.backward()
            optimizer_private.step()
            total_loss += float(private_loss.item()) * batch_x.size(0)
            total_samples += int(batch_x.size(0))
            preds = torch.argmax(private_logits, dim=1)
            correct += int((preds == batch_y).sum().item())
            steps += 1

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, total_samples, steps, accuracy
