import torch
from torch.utils.data import DataLoader
from mrcnn.Prepare_Train_Dataset import train_ds, get_val_ds
from mrcnn.Model import get_model

def collate_fn(batch):
    return tuple(zip(*batch))


# DataLoaders
train_ds = train_ds()
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

val_ds = get_val_ds()
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Optimizer & LR scheduler
model= get_model()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ——— TRAINING ——————————————————————————————————————————————————————————————————————————————————————
    model.train()
    running_train_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_train_loss += losses.item()

        train_loss = running_train_loss / len(train_loader)

    # ——— VALIDATION ————————————————————————————————————————————————————————————————————————————————————
    model.train()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            val_losses = sum(loss for loss in loss_dict.values())
            running_val_loss += val_losses.item()

    val_loss = running_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}  "
          f"Train Loss: {train_loss:.4f}  "
          f"Val Loss: {val_loss:.4f}")

    # ——— SAVE BEST ——————————————————————————————————————————————————————————————————————————————————————
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_maskrcnn.pth")
        print(f"  🎉 New best model saved (val_loss {val_loss:.4f})")


    lr_scheduler.step()
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")
