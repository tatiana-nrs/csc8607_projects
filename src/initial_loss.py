import math, yaml, torch
from src.data_loading import get_dataloaders
from src.model import build_model

# 1) Charger config
cfg = yaml.safe_load(open("configs/config.yaml"))

# 2) Device
if cfg["train"]["device"] == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(cfg["train"]["device"])

# 3) Data + model
train_loader, val_loader, _, meta = get_dataloaders(cfg)
model = build_model(cfg).to(device)
model.train()

# 4) Un batch
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)  # y doit être int64

# 5) Forward + loss
criterion = torch.nn.CrossEntropyLoss()
logits = model(x)  # (B,200)
loss = criterion(logits, y)

print("batch shape:", tuple(x.shape), "labels:", tuple(y.shape))
print("logits shape:", tuple(logits.shape))
print("loss initiale observée:", float(loss.item()))
print("loss attendue ~ log(200) =", math.log(meta["num_classes"]))

# 6) Backward OK ? gradients non nuls ?
model.zero_grad(set_to_none=True)
loss.backward()

p = next(model.parameters())
print("grad abs mean (premier param):", float(p.grad.abs().mean()))
