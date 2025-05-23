# Dataset Distillation
Compressing entire datasets into a small number of synthetic images that can train models to achieve surprisingly good performance.
This is implementation of [DATASET DISTILLATION](https://arxiv.org/pdf/1811.10959)

## Dataset Distillation
### `DatasetDistillation` 
The main class that implements the distillation algorithm
- Supports both fixed and random initialization modes
- Handles multiple gradient steps and epochs
- Implements the bi-level optimization from the paper

#### Key Methods
- `distill()`: Main algorithm that optimizes synthetic images
- `_compute_distillation_loss()`: Implements the loss function from Equation 3
- `sample_initial_weights()`: Handles different initialization strategies
- `visualize_distilled_images()`: Shows the learned synthetic images

#### Algorithm Implementation
- Creates synthetic images with gradient tracking
- Optimizes both the images and learning rate η
- Supports multiple gradient descent steps as described in Section 3.4
- Handles random initialization sampling for better generalization

## Image Distillation 
#### `ImageFolderDataset`
Custom dataset class that:
- Loads images from folders 1/ to 12/
- Handles JPG images
- Converts labels (folder 1 → label 0, folder 12 → label 11)
- Applies appropriate transformations
 
#### `create_data_loaders` 
Function that
- Splits data into train/test sets (80/20 by default)
- Applies data augmentation to training set
- Returns ready-to-use DataLoaders

#### `ConvNetRGB`
A CNN architecture adapted for
- RGB images (3 channels)
- 12 classes
- Configurable image size

### `run_dataset_distillation`
Complete pipeline that
- Loads your data
- Runs the distillation process
- Saves results and visualizations
- Evaluates the distilled images
- Expected directory structure
```
your_data_directory/
├── 1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 2/
│   ├── image1.jpg
│   └── ...
...
└── 12/
    ├── image1.jpg
    └── ...
```

```python
# Simple usage
train_loader, test_loader = create_data_loaders(
    root_dir="/path/to/your/folders",  # Parent directory of 1/, 2/, ..., 12/
    batch_size=256,
    image_size=(32, 32)
)

# Or run the complete distillation
run_dataset_distillation(
    data_dir="/path/to/your/folders",
    images_per_class=1,  # Creates 12 distilled images total
    distillation_steps=1000
)
```

#### Features
- Automatic image preprocessing: Resizing, normalization, and augmentation
- Memory efficient: Uses DataLoader with multiple workers
- GPU support: Automatically uses CUDA if available
- Results saving: Saves distilled images, visualizations, and performance metrics

## OHLC distilation 
#### `OHLCDataset`
Handles time series data with:
- Support for multiple tokens
- Sequence creation for time series prediction
- Both regression (price prediction) and classification (direction prediction)
- Data normalization per token

#### `OHLCDistillation`
Core distillation algorithm adapted for
- Time series sequences instead of images
- OHLC constraints (High ≥ Open, Close, Low)
- Multiple tokens with different patterns
- Temporal dependencies

#### Model Architectures
- LSTM: For capturing temporal patterns
- Transformer: For more complex temporal relationships

#### Special Features for OHLC
- Enforces financial constraints (High/Low boundaries)
- Handles volume data
- Visualizes as candlestick-style charts
- Supports both price prediction and direction classification

#### Key Advantages for Financial Data
- Token-Specific Patterns: Each token gets its own distilled sequence capturing its unique behavior
- Temporal Consistency: Maintains time series properties
- Financial Constraints: Respects OHLC relationships
- Multi-Task Support: Can distill for price prediction or direction classification

```python
# 1. Prepare your data paths
data_paths = {
    'BTC': 'btc_ohlc.csv',
    'ETH': 'eth_ohlc.csv',
    'SOL': 'sol_ohlc.csv',
}

# 2. Run distillation
run_ohlc_distillation(
    data_paths=data_paths,
    sequence_length=30,          # 30 time steps as input
    sequences_per_token=1,       # 1 distilled sequence per token
    distillation_steps=1000,
    model_type='lstm',           # or 'transformer'
    task='regression'            # or 'classification'
)
```


Expected CSV Format
```csv
date,open,high,low,close,volume
2024-01-01,45000,46000,44000,45500,1000000
2024-01-02,45500,47000,45000,46500,1200000
```

Instead of distilling thousands of OHLC bars into a few synthetic images, this
- Creates synthetic OHLC sequences that capture the essential patterns
- Each token gets one (or more) representative sequence
- These sequences can train models to recognize patterns almost as well as the full dataset

This is particularly useful for
- Fast model prototyping
- Privacy-preserving data sharing (synthetic sequences don't reveal actual prices)
- Understanding what patterns are most important for each token
- Quick model adaptation to new tokens

The distilled sequences will visually show the characteristic patterns of each token (e.g., BTC's volatility patterns, ETH's correlation patterns, etc.).

