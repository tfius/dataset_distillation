import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path


class ImageFolderDataset(Dataset):
    """
    Custom dataset for loading images from folders named 1/ to 12/
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (32, 32)
    ):
        """
        Args:
            root_dir: Root directory containing folders 1/ to 12/
            transform: Optional transform to apply to images
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # If no transform provided, create a default one
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Scan folders 1 to 12
        for class_idx in range(1, 13):
            class_folder = self.root_dir / str(class_idx)
            
            if not class_folder.exists():
                print(f"Warning: Folder {class_folder} does not exist")
                continue
            
            # Get all image files in the folder
            for img_file in class_folder.glob("*.jpg"):
                self.image_paths.append(str(img_file))
                # Labels are 0-indexed (folder 1 -> label 0, etc.)
                self.labels.append(class_idx - 1)
        
        print(f"Found {len(self.image_paths)} images across {len(set(self.labels))} classes")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def create_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (32, 32),
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders from the image folders
    
    Args:
        root_dir: Root directory containing folders 1/ to 12/
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, test_loader
    """
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = ImageFolderDataset(root_dir, transform=None, image_size=image_size)
    
    # Split into train and test
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    
    # Create random split
    train_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, test_size]
    )
    
    # Create train dataset with augmentation
    train_dataset = ImageFolderDataset(root_dir, transform=transform_train, image_size=image_size)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    
    # Create test dataset without augmentation
    test_dataset = ImageFolderDataset(root_dir, transform=transform_test, image_size=image_size)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_data_statistics(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and std of the dataset for normalization
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std


# Modified ConvNet for RGB images
class ConvNetRGB(torch.nn.Module):
    """ConvNet adapted for RGB images with 12 classes"""
    def __init__(self, num_classes=12, image_size=32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)  # RGB input
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        
        # Calculate size after convolutions
        conv_output_size = (image_size // 8) * (image_size // 8) * 256
        
        self.fc1 = torch.nn.Linear(conv_output_size, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Complete example of using Dataset Distillation with your data
def run_dataset_distillation(
    data_dir: str,
    save_dir: str = "./distilled_results",
    image_size: int = 32,
    images_per_class: int = 1,
    distillation_steps: int = 1000,
    num_gradient_steps: int = 10,
    num_epochs: int = 3
):
    """
    Run dataset distillation on your image dataset
    
    Args:
        data_dir: Directory containing folders 1/ to 12/
        save_dir: Directory to save results
        image_size: Size to resize images to
        images_per_class: Number of distilled images per class
        distillation_steps: Number of optimization steps
        num_gradient_steps: Number of gradient steps to unroll
        num_epochs: Number of epochs per optimization step
    """
    
    # Import the DatasetDistillation class from previous implementation
    from dataset_distillation import DatasetDistillation
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, test_loader = create_data_loaders(
        root_dir=data_dir,
        batch_size=256,
        image_size=(image_size, image_size),
        train_split=0.8
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset distillation
    print("\nInitializing dataset distillation...")
    distiller = DatasetDistillation(
        num_classes=12,
        image_shape=(3, image_size, image_size),  # RGB images
        images_per_class=images_per_class,
        initialization='random',
        device=device
    )
    
    # Create model function
    def create_model():
        return ConvNetRGB(num_classes=12, image_size=image_size).to(device)
    
    # Run distillation
    print("\nStarting distillation process...")
    distiller.distill(
        model_fn=create_model,
        train_loader=train_loader,
        steps=distillation_steps,
        num_gradient_steps=num_gradient_steps,
        num_epochs=num_epochs,
        lr=0.001
    )
    
    # Get distilled dataset
    distilled_images, distilled_labels = distiller.get_distilled_dataset()
    
    # Save distilled images
    torch.save({
        'distilled_images': distilled_images,
        'distilled_labels': distilled_labels,
        'learning_rate': distiller.eta.item()
    }, os.path.join(save_dir, 'distilled_data.pt'))
    
    # Visualize distilled images
    distiller.visualize_distilled_images(
        save_path=os.path.join(save_dir, 'distilled_images.png')
    )
    
    # Evaluate distilled data
    print("\nEvaluating distilled data...")
    from dataset_distillation_impl import evaluate_distilled_data
    
    mean_acc, std_acc = evaluate_distilled_data(
        distilled_images=distilled_images,
        distilled_labels=distilled_labels,
        test_loader=test_loader,
        model_fn=create_model,
        learning_rate=distiller.eta.item(),
        steps=100,
        num_evaluations=10
    )
    
    print(f"\nDistilled data performance: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Save results
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(f"Dataset Distillation Results\n")
        f.write(f"===========================\n")
        f.write(f"Number of classes: 12\n")
        f.write(f"Images per class: {images_per_class}\n")
        f.write(f"Total distilled images: {12 * images_per_class}\n")
        f.write(f"Distillation steps: {distillation_steps}\n")
        f.write(f"Test accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Learned learning rate: {distiller.eta.item():.4f}\n")


# Example usage
if __name__ == "__main__":
    # Set your data directory path
    data_directory = "/path/to/your/image/folders"  # Update this path
    
    # Run dataset distillation
    run_dataset_distillation(
        data_dir=data_directory,
        save_dir="./distilled_results",
        image_size=32,
        images_per_class=1,  # 12 total images (1 per class)
        distillation_steps=1000,
        num_gradient_steps=10,
        num_epochs=3
    )
