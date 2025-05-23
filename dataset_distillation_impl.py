import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class DatasetDistillation:
    """
    Implementation of Dataset Distillation from Wang et al. 2018
    
    This class implements the core algorithm for distilling a large dataset
    into a small number of synthetic images.
    """
    
    def __init__(
        self,
        num_classes: int,
        image_shape: Tuple[int, int, int],
        images_per_class: int = 1,
        initialization: str = 'random',  # 'random' or 'fixed'
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            num_classes: Number of classes in the dataset
            image_shape: Shape of images (C, H, W)
            images_per_class: Number of distilled images per class
            initialization: Type of network initialization to optimize for
            device: Device to run computations on
        """
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.images_per_class = images_per_class
        self.initialization = initialization
        self.device = device
        
        # Initialize distilled images
        self.distilled_images = self._initialize_distilled_images()
        self.distilled_labels = self._create_labels()
        
        # Initialize learning rate for distilled data
        self.eta = torch.tensor(0.01, requires_grad=True, device=device)
        
    def _initialize_distilled_images(self) -> torch.Tensor:
        """Initialize synthetic distilled images"""
        total_images = self.num_classes * self.images_per_class
        # Initialize with random noise
        images = torch.randn(
            total_images, *self.image_shape, 
            requires_grad=True, 
            device=self.device
        )
        return images
    
    def _create_labels(self) -> torch.Tensor:
        """Create labels for distilled images"""
        labels = []
        for c in range(self.num_classes):
            labels.extend([c] * self.images_per_class)
        return torch.tensor(labels, device=self.device)
    
    def sample_initial_weights(self, model: nn.Module, init_type: str = 'xavier'):
        """Sample initial weights for the model"""
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    if init_type == 'xavier':
                        nn.init.xavier_uniform_(module.weight)
                    elif init_type == 'he':
                        nn.init.kaiming_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.Conv2d):
                    if init_type == 'xavier':
                        nn.init.xavier_uniform_(module.weight)
                    elif init_type == 'he':
                        nn.init.kaiming_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.zero_()
    
    def distill(
        self,
        model_fn,
        train_loader: DataLoader,
        steps: int = 1000,
        num_gradient_steps: int = 1,
        num_epochs: int = 1,
        batch_size: int = 256,
        lr: float = 0.001,
        sample_init_weights: int = 4
    ):
        """
        Main distillation algorithm
        
        Args:
            model_fn: Function that returns a new model instance
            train_loader: DataLoader for the original training data
            steps: Number of optimization steps
            num_gradient_steps: Number of gradient steps to unroll
            num_epochs: Number of epochs to train for
            batch_size: Batch size for sampling real data
            lr: Learning rate for optimizing distilled data
            sample_init_weights: Number of random initializations per step
        """
        
        # Optimizer for distilled images and learning rate
        optimizer = optim.Adam([self.distilled_images, self.eta], lr=lr)
        
        for step in range(steps):
            # Sample a batch of real training data
            real_data, real_labels = next(iter(train_loader))
            real_data = real_data.to(self.device)
            real_labels = real_labels.to(self.device)
            
            total_loss = 0.0
            
            # For random initialization, sample multiple initializations
            if self.initialization == 'random':
                for _ in range(sample_init_weights):
                    # Create new model with random initialization
                    model = model_fn().to(self.device)
                    self.sample_initial_weights(model)
                    
                    # Compute loss
                    loss = self._compute_distillation_loss(
                        model, real_data, real_labels, 
                        num_gradient_steps, num_epochs
                    )
                    total_loss += loss
                
                total_loss /= sample_init_weights
            
            else:  # Fixed initialization
                model = model_fn().to(self.device)
                loss = self._compute_distillation_loss(
                    model, real_data, real_labels,
                    num_gradient_steps, num_epochs
                )
                total_loss = loss
            
            # Update distilled images and learning rate
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Clamp distilled images to valid range
            with torch.no_grad():
                self.distilled_images.clamp_(-3, 3)
            
            if step % 100 == 0:
                print(f"Step {step}/{steps}, Loss: {total_loss.item():.4f}")
    
    def _compute_distillation_loss(
        self,
        model: nn.Module,
        real_data: torch.Tensor,
        real_labels: torch.Tensor,
        num_gradient_steps: int,
        num_epochs: int
    ) -> torch.Tensor:
        """
        Compute the distillation loss by:
        1. Training model on distilled data for specified steps/epochs
        2. Evaluating the trained model on real data
        """
        
        # Clone model to avoid modifying the original
        model_copy = self._clone_model_params(model)
        
        # Train on distilled data
        for epoch in range(num_epochs):
            for step in range(num_gradient_steps):
                # Forward pass on distilled data
                outputs = model_copy(self.distilled_images)
                loss = F.cross_entropy(outputs, self.distilled_labels)
                
                # Compute gradients w.r.t model parameters
                grads = torch.autograd.grad(
                    loss, model_copy.parameters(), create_graph=True
                )
                
                # Update model parameters (gradient descent)
                with torch.no_grad():
                    for param, grad in zip(model_copy.parameters(), grads):
                        param.sub_(self.eta * grad)
        
        # Evaluate on real data
        outputs = model_copy(real_data)
        real_loss = F.cross_entropy(outputs, real_labels)
        
        return real_loss
    
    def _clone_model_params(self, model: nn.Module) -> nn.Module:
        """Create a copy of model with cloned parameters"""
        model_copy = type(model)().to(self.device)
        model_copy.load_state_dict(model.state_dict())
        
        # Make parameters require gradients for backpropagation
        for param in model_copy.parameters():
            param.requires_grad = True
            
        return model_copy
    
    def get_distilled_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the distilled images and labels"""
        return self.distilled_images.detach(), self.distilled_labels
    
    def visualize_distilled_images(self, save_path: Optional[str] = None):
        """Visualize the distilled images"""
        images = self.distilled_images.detach().cpu()
        
        # Normalize images for visualization
        images = (images - images.min()) / (images.max() - images.min())
        
        # Create grid
        fig, axes = plt.subplots(
            self.num_classes, self.images_per_class,
            figsize=(self.images_per_class * 2, self.num_classes * 2)
        )
        
        if self.num_classes == 1:
            axes = axes.reshape(1, -1)
        if self.images_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        idx = 0
        for c in range(self.num_classes):
            for i in range(self.images_per_class):
                img = images[idx]
                if img.shape[0] == 1:  # Grayscale
                    axes[c, i].imshow(img[0], cmap='gray')
                else:  # RGB
                    axes[c, i].imshow(img.permute(1, 2, 0))
                axes[c, i].set_title(f'Class {c}')
                axes[c, i].axis('off')
                idx += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# Example usage with a simple ConvNet
class SimpleConvNet(nn.Module):
    """Simple ConvNet for MNIST/CIFAR10"""
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_distilled_data(
    distilled_images: torch.Tensor,
    distilled_labels: torch.Tensor,
    test_loader: DataLoader,
    model_fn,
    learning_rate: float,
    steps: int = 100,
    num_evaluations: int = 10
) -> float:
    """
    Evaluate the quality of distilled data by training new models on it
    and testing on real test data
    """
    accuracies = []
    
    for _ in range(num_evaluations):
        # Create new model with random initialization
        model = model_fn()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        # Train on distilled data
        for step in range(steps):
            outputs = model(distilled_images)
            loss = F.cross_entropy(outputs, distilled_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)


# Example of how to use the implementation
if __name__ == "__main__":
    # Example for MNIST-like data
    num_classes = 10
    image_shape = (1, 28, 28)  # MNIST shape
    images_per_class = 1  # 10 images total
    
    # Initialize dataset distillation
    distiller = DatasetDistillation(
        num_classes=num_classes,
        image_shape=image_shape,
        images_per_class=images_per_class,
        initialization='random'
    )
    
    # Create model function
    def create_model():
        return SimpleConvNet(num_classes=num_classes, input_channels=1)
    
    # Note: You would need to provide your actual train_loader here
    # train_loader = DataLoader(your_dataset, batch_size=256, shuffle=True)
    
    # Run distillation (commented out as it requires actual data)
    # distiller.distill(
    #     model_fn=create_model,
    #     train_loader=train_loader,
    #     steps=1000,
    #     num_gradient_steps=10,
    #     num_epochs=3
    # )
    
    # Get distilled dataset
    # distilled_images, distilled_labels = distiller.get_distilled_dataset()
    
    # Visualize results
    # distiller.visualize_distilled_images('distilled_images.png')
