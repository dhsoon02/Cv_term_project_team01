import torch, timm, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import csv

class ActionDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path

class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return ImageOps.expand(image, padding, fill=(0, 0, 0))

train_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# Prepare CSV to record per-image results
csv_filename = "mobilenet_kfold_image_scores.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fold', 'image_path', 'true_label', 'predicted_label', 'correct'])

# K-Fold loop
for fold in range(5):
    print(f"\n========= Fold {fold} =========")

    train_dataset = ActionDataset(f"action_crops_kfold/fold{fold + 1}/train.txt", transform=train_transform)
    val_dataset = ActionDataset(f"action_crops_kfold/fold{fold + 1}/val.txt", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=4)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for images, labels, _ in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        per_image_results = []
        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, labels)
                total_val_loss += loss.item()
                preds = output.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for path, pred, label in zip(paths, preds.cpu(), labels.cpu()):
                    per_image_results.append([fold, path, label.item(), pred.item(), int(pred == label)])

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}: Train Loss={total_train_loss:.4f} | Val Loss={total_val_loss:.4f} | Val Acc={val_accuracy:.4f}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f"mobilenetv3_fold{fold}_best.pth")
            print("✅ Best model updated (acc)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    # Append fold results to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(per_image_results)

print("\n✅ K-Fold training complete. Results saved to:", csv_filename)