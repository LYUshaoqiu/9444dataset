import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, mobilenet_v2, vgg16, vit_b_16, resnet50
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from models.vgg_origin import vgg_origin
# from models.vgg_attention import vgg_attention
from models.vgg_attention_dropout import vgg_attention
from models.vgg_multi_head_attention import vgg_multi_head_attention
from collections import Counter

class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        folders = ['bags', 'clothes', 'accessories', 'shoes']
        for label, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            for file_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class ComplexDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        folders = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
        for label, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            for file_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def main(args):
    if args.transform == 'simple':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif args.transform == 'complex':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    # load dataset
    if args.dataset == 'simple':
        dataset = SimpleDataset(root_dir='./datasets/simple', transform=transform)

        dataset_size = len(dataset)
        train_size = int(0.6 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # dataloader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

        # set num class
        num_class = 4

    elif args.dataset == 'complex':
        train_dataset = ComplexDataset(root_dir='./datasets/complex/train', transform=transform)
        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        class_counts = Counter(train_dataset.labels)
        total_samples = sum(class_counts.values())
        class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))]).to(device)

        # Assign weights to each sample
        sample_weights = [class_weights[label] for label in train_dataset.labels]

        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Use the sampler in the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

        val_dataset = ComplexDataset(root_dir='./datasets/complex/validation', transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        test_dataset = ComplexDataset(root_dir='./datasets/complex/test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # set num class
        num_class = 10

    # load model
    if args.model == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=args.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_class)
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=args.pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_class)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif args.model == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
        model.classifier[6] = nn.Linear(4096, num_class)
    elif args.model == 'vit_b_16':
        model = vit_b_16(pretrained=args.pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif args.model == 'vgg_origin':
        print('Use the origin vgg')
        print('No batch norm')
        model = vgg_origin(num_classes=num_class)
    elif args.model == 'vgg_attention':
        print('Use the vgg with attention')
        model = vgg_attention(num_classes=num_class, init_weights=True, batch_norm=args.batch_norm)
    elif args.model == 'vgg_multi_head_attention':
        print('Use the vgg with multi-head attention')
        model = vgg_multi_head_attention(num_classes=num_class, init_weights=True, batch_norm=args.batch_norm)

    # use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_class)]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_accuracy = 0.0
    best_model_weights = model.state_dict()
    patience = args.patience
    no_improvement_count = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=running_loss/(train_total/32), accuracy=train_correct/train_total)
        
        # add loss and accuracy for plot
        train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_bar = tqdm(val_loader, desc=f'Validating Epoch {epoch+1}')
        
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix(accuracy=val_correct/val_total)
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')
        
        # save the best model validation
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            print("Early stopping triggered")
            break

    # load best weights
    model.load_state_dict(best_model_weights)

    # test
    model.eval()
    correct = 0
    total = 0
    test_bar = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_bar.set_postfix(accuracy=correct/total)

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_losses) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'./plots/{args.model}_{args.dataset}.png')
    plt.close()
    
    # save config accuracy logs
    with open(f'./logs/{args.model}_{args.dataset}.log', 'a') as file:
        file.write(str(args) + '\n')
        file.write(f'Valid Accuracy: {best_val_accuracy}, Test Accuracy: {test_accuracy}' + '\n')
        file.write('==============================================\n\n')
        print("Logs Write Successful")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='complex')
    parser.add_argument('--model', type=str, default='vgg_origin')
    parser.add_argument('--pretrained', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--batch_norm', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--transform', type=str, default='simple')
    args = parser.parse_args()
    print(args)
    main(args)
