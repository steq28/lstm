# LSTM News Headline Generator

A PyTorch implementation of a Language Model using LSTM (Long Short-Term Memory) networks to generate political news headlines. The model is trained on the Hugging Face `heegyu/news-category-dataset` dataset, specifically focusing on headlines from the POLITICS category.

## Features

- LSTM-based language model with embedding layer
- Two training implementations:
  - Standard training
  - Truncated Backpropagation Through Time (TBPTT)
- Two text generation strategies:
  - Random sampling with top-k
  - Greedy (argmax) sampling
- Performance visualization with loss and perplexity plots

## Requirements

```
torch
datasets
matplotlib
numpy
```

## Dataset

The project uses the `heegyu/news-category-dataset` from Hugging Face, filtering for political headlines. The data processing pipeline includes:

- Lowercase conversion
- Basic tokenization
- Addition of `<EOS>` tokens
- Creation of word-to-index and index-to-word mappings
- Padding for batch processing

## Model Architecture

The LSTM model includes:

- Embedding layer (configurable dimension)
- LSTM layer(s) (configurable number of layers and hidden size)
- Fully connected output layer
- Dropout for regularization

## Usage

1. Clone the repository:
```bash
git clone https://github.com/steq28/lstm-news-generator
cd lstm-news-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training:
```bash
python lstm-model.py
```

## Model Parameters

Default hyperparameters:
- Hidden size: 1024 (standard) / 2048 (truncated)
- Embedding dimension: 150
- Number of LSTM layers: 1
- Learning rate: 0.001
- Batch size: 32
- Training epochs: 6
- Gradient clipping: 1.0

## Results

The model achieves:
- Loss < 1.5 by epoch 6 in standard training
- Loss < 0.9 by epoch 6 in truncated training
- Generates coherent political headlines with both sampling strategies

Example outputs:
```
Random sampling:
- "the president wants a letter to foreign and justice <EOS>"
- "the president wants to help the other <EOS>"
- "the president wants a money advantage in american politics <EOS>"

Greedy sampling:
- "the president wants to help the koch brothers <EOS>"
```

## Visualizations

The training process generates four plots:
- Training loss (standard training)
- Perplexity (standard training)
- Training loss (truncated training)
- Perplexity (truncated training)

## Implementation Details

### Key Components

1. `Dataset` class: Handles data preprocessing and batching
2. `Model` class: Implements the LSTM architecture
3. Training functions:
   - `train()`: Standard training implementation
   - `train_truncated()`: TBPTT implementation
4. Generation functions:
   - `random_sample_next()`: Top-k random sampling
   - `sample_argmax()`: Greedy sampling

### Note on Embeddings

Unlike Word2Vec, this model uses contextual embeddings, meaning the vector representations depend on the surrounding context and don't maintain static arithmetic properties (e.g., King - Man + Woman ≠ Queen).

## Author

Stefano Quaggio (stefano.quaggio@usi.ch)
