import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json


class OHLCDataset(Dataset):
    """Dataset for OHLC time series data from multiple tokens"""
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        features: List[str] = ['open', 'high', 'low', 'close', 'volume'],
        target_feature: str = 'close',
        task: str = 'regression',  # 'regression' or 'classification'
        classification_threshold: float = 0.0  # For binary classification (up/down)
    ):
        """
        Args:
            data_dict: Dictionary mapping token names to DataFrames with OHLC data
            sequence_length: Number of time steps to use as input
            prediction_horizon: Number of steps ahead to predict
            features: List of features to use from OHLC data
            target_feature: Feature to predict
            task: 'regression' for price prediction, 'classification' for direction
            classification_threshold: Threshold for binary classification
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.target_feature = target_feature
        self.task = task
        self.classification_threshold = classification_threshold
        
        # Process all tokens
        self.sequences = []
        self.targets = []
        self.token_labels = []
        self.scalers = {}
        
        for token_idx, (token_name, df) in enumerate(data_dict.items()):
            # Normalize data per token
            normalized_df, scaler = self._normalize_data(df[features])
            self.scalers[token_name] = scaler
            
            # Create sequences
            for i in range(len(normalized_df) - sequence_length - prediction_horizon + 1):
                seq = normalized_df[i:i + sequence_length].values
                
                if task == 'regression':
                    # Predict actual value
                    target_idx = self.features.index(target_feature)
                    target = normalized_df.iloc[i + sequence_length + prediction_horizon - 1, target_idx]
                else:
                    # Predict direction (up/down)
                    current_price = df[target_feature].iloc[i + sequence_length - 1]
                    future_price = df[target_feature].iloc[i + sequence_length + prediction_horizon - 1]
                    returns = (future_price - current_price) / current_price
                    target = 1 if returns > classification_threshold else 0
                
                self.sequences.append(seq)
                self.targets.append(target)
                self.token_labels.append(token_idx)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.token_labels = np.array(self.token_labels, dtype=np.int64)
        
    def _normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize data using min-max scaling"""
        scaler = {}
        normalized_df = df.copy()
        
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            scaler[col] = {'min': min_val, 'max': max_val}
            
            if max_val > min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0.5
        
        return normalized_df, scaler
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32 if self.task == 'regression' else torch.long),
            torch.tensor(self.token_labels[idx], dtype=torch.long)
        )


class OHLCDistillation:
    """Dataset Distillation for OHLC time series data"""
    
    def __init__(
        self,
        num_tokens: int,
        sequence_length: int = 30,
        num_features: int = 5,
        sequences_per_token: int = 1,
        task: str = 'regression',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            num_tokens: Number of different tokens in the dataset
            sequence_length: Length of time series sequences
            num_features: Number of features (OHLC + volume = 5)
            sequences_per_token: Number of distilled sequences per token
            task: 'regression' or 'classification'
            device: Device to run computations on
        """
        self.num_tokens = num_tokens
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.sequences_per_token = sequences_per_token
        self.task = task
        self.device = device
        
        # Initialize distilled sequences
        self.distilled_sequences = self._initialize_distilled_sequences()
        self.distilled_targets = self._initialize_distilled_targets()
        self.distilled_token_labels = self._create_token_labels()
        
        # Initialize learning rate
        self.eta = torch.tensor(0.01, requires_grad=True, device=device)
        
    def _initialize_distilled_sequences(self) -> torch.Tensor:
        """Initialize synthetic OHLC sequences"""
        total_sequences = self.num_tokens * self.sequences_per_token
        
        # Initialize with reasonable values for normalized OHLC data
        sequences = torch.rand(
            total_sequences, 
            self.sequence_length, 
            self.num_features,
            requires_grad=True,
            device=self.device
        )
        
        # Ensure OHLC constraints (High >= Open, Close, Low)
        with torch.no_grad():
            # Assuming features are [open, high, low, close, volume]
            if self.num_features >= 4:
                sequences[:, :, 1] = sequences[:, :, [0, 2, 3]].max(dim=2)[0] + 0.01  # High
                sequences[:, :, 2] = sequences[:, :, [0, 1, 3]].min(dim=2)[0] - 0.01  # Low
        
        return sequences
    
    def _initialize_distilled_targets(self) -> torch.Tensor:
        """Initialize targets for distilled sequences"""
        total_sequences = self.num_tokens * self.sequences_per_token
        
        if self.task == 'regression':
            # Initialize with mid-range values
            targets = torch.rand(total_sequences, requires_grad=True, device=self.device)
        else:
            # For classification, initialize with random classes
            targets = torch.randint(0, 2, (total_sequences,), device=self.device).float()
            targets.requires_grad = True
        
        return targets
    
    def _create_token_labels(self) -> torch.Tensor:
        """Create token labels for distilled sequences"""
        labels = []
        for token in range(self.num_tokens):
            labels.extend([token] * self.sequences_per_token)
        return torch.tensor(labels, device=self.device)
    
    def enforce_ohlc_constraints(self):
        """Enforce OHLC constraints on distilled sequences"""
        with torch.no_grad():
            if self.num_features >= 4:
                # High should be >= Open, Close, Low
                max_vals = torch.max(
                    torch.stack([
                        self.distilled_sequences[:, :, 0],  # Open
                        self.distilled_sequences[:, :, 2],  # Low
                        self.distilled_sequences[:, :, 3]   # Close
                    ], dim=0), 
                    dim=0
                )[0]
                self.distilled_sequences[:, :, 1] = torch.maximum(
                    self.distilled_sequences[:, :, 1], 
                    max_vals + 0.001
                )
                
                # Low should be <= Open, Close, High
                min_vals = torch.min(
                    torch.stack([
                        self.distilled_sequences[:, :, 0],  # Open
                        self.distilled_sequences[:, :, 1],  # High
                        self.distilled_sequences[:, :, 3]   # Close
                    ], dim=0), 
                    dim=0
                )[0]
                self.distilled_sequences[:, :, 2] = torch.minimum(
                    self.distilled_sequences[:, :, 2], 
                    min_vals - 0.001
                )
            
            # Clamp all values to [0, 1] range (normalized)
            self.distilled_sequences.clamp_(0, 1)
            
            # Ensure volume is positive
            if self.num_features >= 5:
                self.distilled_sequences[:, :, 4].clamp_(min=0)
    
    def distill(
        self,
        model_fn,
        train_loader: DataLoader,
        steps: int = 1000,
        num_gradient_steps: int = 10,
        lr: float = 0.001,
        sample_init_weights: int = 4
    ):
        """
        Main distillation algorithm adapted for OHLC data
        """
        # Optimizer for distilled sequences and learning rate
        optimizer = optim.Adam([self.distilled_sequences, self.distilled_targets, self.eta], lr=lr)
        
        for step in range(steps):
            # Sample a batch of real training data
            real_sequences, real_targets, real_token_labels = next(iter(train_loader))
            real_sequences = real_sequences.to(self.device)
            real_targets = real_targets.to(self.device)
            
            total_loss = 0.0
            
            # Sample multiple random initializations
            for _ in range(sample_init_weights):
                # Create new model with random initialization
                model = model_fn().to(self.device)
                
                # Initialize weights
                for module in model.modules():
                    if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU)):
                        for name, param in module.named_parameters():
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                param.data.zero_()
                
                # Compute loss
                loss = self._compute_distillation_loss(
                    model, real_sequences, real_targets, num_gradient_steps
                )
                total_loss += loss
            
            total_loss /= sample_init_weights
            
            # Update distilled data
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Enforce OHLC constraints
            self.enforce_ohlc_constraints()
            
            if step % 100 == 0:
                print(f"Step {step}/{steps}, Loss: {total_loss.item():.4f}")
    
    def _compute_distillation_loss(
        self,
        model: nn.Module,
        real_sequences: torch.Tensor,
        real_targets: torch.Tensor,
        num_gradient_steps: int
    ) -> torch.Tensor:
        """Compute distillation loss for time series"""
        # Clone model parameters
        model_copy = self._clone_model(model)
        
        # Train on distilled data
        for step in range(num_gradient_steps):
            outputs = model_copy(self.distilled_sequences)
            
            if self.task == 'regression':
                loss = F.mse_loss(outputs.squeeze(), self.distilled_targets)
            else:
                loss = F.cross_entropy(outputs, self.distilled_targets.long())
            
            # Compute gradients
            grads = torch.autograd.grad(loss, model_copy.parameters(), create_graph=True)
            
            # Update model parameters
            with torch.no_grad():
                for param, grad in zip(model_copy.parameters(), grads):
                    param.sub_(self.eta * grad)
        
        # Evaluate on real data
        outputs = model_copy(real_sequences)
        
        if self.task == 'regression':
            real_loss = F.mse_loss(outputs.squeeze(), real_targets)
        else:
            real_loss = F.cross_entropy(outputs, real_targets)
        
        return real_loss
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of model with cloned parameters"""
        model_copy = type(model)().to(self.device)
        model_copy.load_state_dict(model.state_dict())
        
        for param in model_copy.parameters():
            param.requires_grad = True
            
        return model_copy
    
    def visualize_distilled_sequences(self, token_names: Optional[List[str]] = None):
        """Visualize the distilled OHLC sequences"""
        sequences = self.distilled_sequences.detach().cpu().numpy()
        
        fig, axes = plt.subplots(
            self.num_tokens, 
            self.sequences_per_token,
            figsize=(5 * self.sequences_per_token, 3 * self.num_tokens)
        )
        
        if self.num_tokens == 1:
            axes = axes.reshape(1, -1)
        if self.sequences_per_token == 1:
            axes = axes.reshape(-1, 1)
        
        idx = 0
        for token in range(self.num_tokens):
            for seq in range(self.sequences_per_token):
                ax = axes[token, seq]
                data = sequences[idx]
                
                # Plot candlestick-style chart
                for t in range(len(data)):
                    o, h, l, c = data[t, :4]
                    color = 'g' if c >= o else 'r'
                    
                    # High-Low line
                    ax.plot([t, t], [l, h], color='k', linewidth=1)
                    # Open-Close box
                    ax.plot([t, t], [o, c], color=color, linewidth=3)
                
                token_name = token_names[token] if token_names else f"Token {token}"
                ax.set_title(f'{token_name} - Seq {seq}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price (normalized)')
                idx += 1
        
        plt.tight_layout()
        plt.show()


# LSTM model for time series
class LSTMModel(nn.Module):
    """LSTM model for OHLC time series prediction"""
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


# Transformer model for time series
class TransformerModel(nn.Module):
    """Transformer model for OHLC time series prediction"""
    
    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(100, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transform
        x = self.transformer(x)
        
        # Use last time step
        output = self.fc(x[:, -1, :])
        return output


def load_ohlc_data(
    data_paths: Dict[str, str],
    date_column: str = 'date',
    ohlc_columns: Dict[str, str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLC data from CSV files
    
    Args:
        data_paths: Dictionary mapping token names to file paths
        date_column: Name of the date column
        ohlc_columns: Mapping of standard names to column names in your data
    
    Returns:
        Dictionary mapping token names to DataFrames
    """
    if ohlc_columns is None:
        ohlc_columns = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
    
    data_dict = {}
    
    for token_name, file_path in data_paths.items():
        df = pd.read_csv(file_path)
        
        # Ensure date parsing
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in ohlc_columns.items()})
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 1000000  # Default volume
                else:
                    df[col] = df['close']  # Use close price as default
        
        data_dict[token_name] = df
    
    return data_dict


# Example usage
def run_ohlc_distillation(
    data_paths: Dict[str, str],
    sequence_length: int = 30,
    sequences_per_token: int = 1,
    distillation_steps: int = 1000,
    model_type: str = 'lstm',  # 'lstm' or 'transformer'
    task: str = 'regression'   # 'regression' or 'classification'
):
    """Run dataset distillation on OHLC data"""
    
    # Load data
    print("Loading OHLC data...")
    data_dict = load_ohlc_data(data_paths)
    token_names = list(data_dict.keys())
    
    # Create dataset
    dataset = OHLCDataset(
        data_dict=data_dict,
        sequence_length=sequence_length,
        task=task
    )
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize distillation
    print("\nInitializing OHLC dataset distillation...")
    distiller = OHLCDistillation(
        num_tokens=len(token_names),
        sequence_length=sequence_length,
        num_features=5,  # OHLCV
        sequences_per_token=sequences_per_token,
        task=task
    )
    
    # Create model function
    def create_model():
        if model_type == 'lstm':
            return LSTMModel(
                input_size=5,
                output_size=1 if task == 'regression' else 2
            )
        else:
            return TransformerModel(
                input_size=5,
                output_size=1 if task == 'regression' else 2
            )
    
    # Run distillation
    print(f"\nRunning distillation with {model_type.upper()} model...")
    distiller.distill(
        model_fn=create_model,
        train_loader=train_loader,
        steps=distillation_steps,
        num_gradient_steps=10
    )
    
    # Visualize results
    print("\nVisualizing distilled sequences...")
    distiller.visualize_distilled_sequences(token_names)
    
    # Save distilled data
    torch.save({
        'sequences': distiller.distilled_sequences.detach().cpu(),
        'targets': distiller.distilled_targets.detach().cpu(),
        'token_labels': distiller.distilled_token_labels.cpu(),
        'token_names': token_names,
        'learning_rate': distiller.eta.item()
    }, 'distilled_ohlc_data.pt')
    
    print("\nDistillation complete!")


if __name__ == "__main__":
    # Example usage
    data_paths = {
        'BTC': 'path/to/btc_ohlc.csv',
        'ETH': 'path/to/eth_ohlc.csv',
        'SOL': 'path/to/sol_ohlc.csv',
        # Add more tokens as needed
    }
    
    run_ohlc_distillation(
        data_paths=data_paths,
        sequence_length=30,
        sequences_per_token=1,
        distillation_steps=1000,
        model_type='lstm',
        task='regression'
    )
