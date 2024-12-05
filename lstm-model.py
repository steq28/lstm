'''
Assignment 3
Stefano Quaggio
'''
import torch
from datasets import load_dataset
from typing import List, Dict, Optional, Any, Tuple, Union
import os
import time
import math
import string
import random

import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.functional import F
from torch.utils.data import DataLoader

# Uncomment this code for using a more advanced tokenizer method
# !pip install spacy
# from spacy import tokenizer
# from spacy.lang.en import English

# nlp = English()
# tkz = tokenizer.Tokenizer(nlp.vocab)

def keys_to_values(keys: List[Any], map: Dict[Any, Any], default_if_missing: Optional[Any] = None):
    """
    Maps a list of keys to their corresponding values in a dictionary.
    
    Input:
    - keys: List of keys to map.
    - map: Dictionary for key-to-value mapping.
    - default_if_missing: Value to use if a key is missing (optional).
    
    Output:
    - List of mapped values.
    """
    return [map.get(key, default_if_missing) for key in keys]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_as_str, map):
        self.data_as_int = []

        # Convert characters to integers
        for seq_as_str in data_as_str:
            seq_as_int = keys_to_values(seq_as_str, map,
                                        random.choice(list(map)))

            self.data_as_int.append(seq_as_int)

    def __len__(self):
        return len(self.data_as_int)

    def __getitem__(self, ix):
        # Get data sample at index ix
        item = self.data_as_int[ix]

        # Slice x and y from sample
        x = item[:-1]
        y = item[ 1:]
        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: int):
    """
    Pads a batch of sequences to the same length.
    
    Input:
    - batch: List of (data, target) tuples of tensors.
    - pad_value: Padding value for sequences.
    
    Output:
    - Tuple of padded data and targets as tensors.
    """

    data, targets = zip(*batch)

    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_value)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)

    return padded_data, padded_targets

def random_sample_next(model: nn.Module, x: torch.Tensor, prev_state: Tuple[torch.Tensor, torch.Tensor], topk: int = 5, uniform: bool = True):
    """
    Samples the next token based on logits with optional uniform or weighted probabilities.
    
    Input:
    - model: The model used for prediction.
    - x: Input tensor for the current step.
    - prev_state: Previous hidden states of the model.
    - topk: Number of top logits to sample from.
    - uniform: Whether to sample uniformly or use softmax probabilities.
    
    Output:
    - The sampled token index and updated state.
    """

    out, state = model(x, prev_state)
    last_out = out[0, -1, :]
    
    topk = topk if topk else last_out.shape[0]
    top_logit, top_ix = torch.topk(last_out, k=topk, dim = -1)
    p = None if uniform else F.softmax(top_logit.detach(), dim=-1).cpu().numpy()
    
    sampled_ix = np.random.choice(top_ix.cpu().numpy(), p=p)
    
    return sampled_ix, state


def sample_argmax(model: nn.Module, x: torch.Tensor, prev_state: Tuple[torch.Tensor, torch.Tensor]):
    """
    Samples the next token by selecting the highest-probability logit.
    
    Input:
    - model: The model used for prediction.
    - x: Input tensor for the current step.
    - prev_state: Previous hidden states of the model.
    
    Output:
    - The sampled token index and updated state.
    """
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]  # Get the logits for the last output step
    
    # Get the index of the word with the highest probability
    sampled_ix = torch.argmax(last_out).item()
    
    return sampled_ix, state



def sample(model: nn.Module, type: str, seed: Union[List[int], int], topk: int = 5, uniform: bool = True, length: int = 100, stop_on: Optional[int] = None) -> List[int]:
    """
    Generates a sequence of tokens from the model using specified sampling type.
    
    Input:
    - model: The model used for prediction.
    - type: Sampling type ('random' or 'greedy').
    - seed: Initial sequence to start generation.
    - topk: Number of top logits to sample from.
    - uniform: Whether to sample uniformly or use softmax probabilities.
    - length: Maximum length of the sequence to generate.
    - stop_on: Token index to stop generation (optional).
    
    Output:
    - List of generated token indices.
    """

    seed = seed if isinstance(seed, (list, tuple)) else [seed]
    model.eval()

    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = torch.tensor([seed]).to(DEVICE)

        prev_state = model.init_state(b_size=1)

        for _ in range(length - len(seed)):
            # Uncomment for random sampling
            if type == "random":
                sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk, uniform)
            else:
                sampled_ix, prev_state = sample_argmax(model, x, prev_state)

            sampled_ix_list.append(sampled_ix)
            x = torch.tensor([[sampled_ix]]).to(DEVICE)
            
            if sampled_ix == stop_on:
                break
        model.train()
    return sampled_ix_list

"""
Question: Model Definition
"""

class Model(nn.Module):
    def __init__(self, map, hidden_size, emb_dim=8, n_layers=1, dropout_p = 0.2):
        super(Model, self).__init__()

        self.vocab_size  = len(map)
        self.hidden_size = hidden_size
        self.emb_dim     = emb_dim
        self.n_layers    = n_layers
        self.dropout_p   = dropout_p

        # dimensions: batches x seq_length x emb_dim
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim =self.emb_dim,
            padding_idx=map["PAD"])

        self.lstm = nn.LSTM(input_size=self.emb_dim,
                          hidden_size=self.hidden_size,
                          num_layers =self.n_layers,
                          batch_first=True)

        self.fc = nn.Linear(
            in_features =self.hidden_size,
            out_features=self.vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        yhat, state = self.lstm(embed, prev_state)   # yhat is the full sequence prediction, while state is the last hidden state (coincides with yhat[-1] if n_layers=1)

        out = self.fc(yhat)
        return out, state

    def init_state(self, b_size=1):
        # Initialize both hidden state (h_0) and cell state (c_0)
        h_0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(DEVICE)
        c_0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(DEVICE)
        return (h_0, c_0)


def print_sentence(model: Model = None, type: str = "greedy", num: int = 100):
    """
    Generates and prints a sentence from the model.
    
    Input:
    - model: The model used for generation.
    - type: Sampling type ('random' or 'greedy').
    - num: Number of tokens to generate.
    
    Output:
    - None (prints the generated sentence).
    """

    seed = keys_to_values(sentence.split(), word_to_int, word_to_int["PAD"])
    sampled_ix_list = sample(model, type, seed, 5, False, num, stop_on=word_to_int["<EOS>"])
    print("Generated sentence:", " ".join(keys_to_values(sampled_ix_list, int_to_word)))
    
def train(model: nn.Module, data: torch.utils.data.DataLoader, num_epochs: int, criterion: nn.Module, lr: float = 0.001, print_every: int = 50, clip: float = None):
    """
    Trains the model on the given dataset.
    
    Input:
    - model: The model to be trained.
    - data: DataLoader providing training data.
    - num_epochs: Number of epochs to train for.
    - criterion: Loss function to use.
    - lr: Learning rate for the optimizer.
    - print_every: Frequency (in epochs) for logging progress.
    - clip: Gradient clipping value (optional).
    
    Output:
    - Trained model.
    - List of per-batch costs.
    - List of average losses per epoch.
    - List of perplexities per epoch.
    """
    model.train()

    costs = []
    running_loss = 0
    loss_hist = []
    perplexity_hist = []

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    while epoch<num_epochs:
        epoch += 1
        for x, y in data:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()
            # Initialise model's state and perform forward-prop
            prev_state = model.init_state(b_size=x.shape[0])
            out, state = model(x, prev_state)         # out has dim: batch x seq_length x vocab_size

            # Calculate loss
            loss = criterion(out.transpose(1, 2), y)  #transpose is required to obtain batch x vocab_size x seq_length
            costs.append(loss.item())
            running_loss += loss.item()

            # Calculate gradients and update parameters
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if print_every and (epoch%print_every)==0:
            avg_loss = running_loss/float(print_every*len(data))
            perplexity = math.exp(avg_loss)
            
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_loss:8.4f}, Perplexity: {perplexity:.4f}")
            loss_hist.append(running_loss/float(print_every*len(data)))
            perplexity_hist.append(perplexity)
            running_loss = 0

        if epoch == 1 or epoch == num_epochs // 2:
            print_sentence(model)

    print_sentence(model)
            

    return model, costs, loss_hist, perplexity_hist


def train_truncated(model_tr: nn.Module, data: torch.utils.data.DataLoader, num_epochs: int, criterion: nn.Module, lr: float = 0.001, print_every: int = 50, clip: float = None, bptt_steps: int = 35):
    """
    Trains the model using truncated backpropagation through time (BPTT).
    
    Input:
    - model_tr: The model to be trained.
    - data: DataLoader providing training data.
    - num_epochs: Number of epochs to train for.
    - criterion: Loss function to use.
    - lr: Learning rate for the optimizer.
    - print_every: Frequency (in epochs) for logging progress.
    - clip: Gradient clipping value (optional).
    - bptt_steps: Number of steps to truncate backpropagation through time.
    
    Output:
    - Trained model.
    - List of per-batch costs.
    - List of average losses per epoch.
    - List of perplexities per epoch.
    """

    model_tr.train()

    # Lists to store metrics
    costs = []
    loss_hist = []
    perplexity_hist = []

    optimizer = optim.Adam(model_tr.parameters(), lr=lr)

    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        total_tokens = 0  # Total tokens processed in the epoch
        total_loss = 0  # Total loss in the epoch
        running_loss = 0  # Accumulated loss for logging

        for x, y in data:
            # Convert to LongTensor for embedding
            x = x.to(DEVICE).long()
            y = y.to(DEVICE).long()

            # Initialize hidden state
            prev_state = model_tr.init_state(b_size=x.shape[0])

            # Process the input sequence in chunks of bptt_steps
            for i in range(0, x.size(1), bptt_steps):
                # Get a chunks
                x_chunk = x[:, i:i + bptt_steps]  
                y_chunk = y[:, i:i + bptt_steps]

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass through the chunk
                out, prev_state = model_tr(x_chunk, prev_state)

                # Detach the hidden state to prevent backpropagating through the entire sequence
                prev_state = tuple(s.detach() for s in prev_state)

                # Compute loss
                loss = criterion(out.transpose(1, 2), y_chunk)
                costs.append(loss.item())
                running_loss += loss.item()

                # Compute total tokens and total loss
                total_tensor_items = y_chunk.numel()  # Number of tokens in y_chunk
                total_tokens += total_tensor_items
                total_loss += loss.item() * total_tensor_items

                # Backpropagation
                loss.backward()

                # Gradient clipping (if specified)
                if clip:
                    nn.utils.clip_grad_norm_(model_tr.parameters(), clip)

                # Update model parameters
                optimizer.step()

        # Perplexity: exp(total_loss / total_tokens)
        if total_tokens > 0:
            perplexity = math.exp(total_loss / total_tokens)
        else:
            perplexity = float('inf')  # Handle edge case

    
        # Average loss per batch
        avg_loss = running_loss / (len(data))
        loss_hist.append(avg_loss)
        perplexity_hist.append(perplexity)

        # Logging
        if print_every and (epoch % print_every) == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_loss:8.4f}, Perplexity: {perplexity:.4f}")
            
        if epoch == 1 or epoch == num_epochs // 2:
            print_sentence(model_tr)
    print_sentence(model_tr)

    return model_tr, costs, loss_hist, perplexity_hist


if __name__ == "__main__":
    # Set the seed
    seed = 42
    torch.manual_seed(seed)
    # Probably, this below must be changed if you work with a M1/M2/M3 Mac
    torch.cuda.manual_seed(seed) # for CUDA
    torch.backends.cudnn.deterministic = True # for CUDNN
    torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')
    
    """
    Question: Data
    """

    ds = load_dataset("heegyu/news-category-dataset")

    print("Dataset: ", ds)
    print("Train columns: ", ds['train'].features)


    politics = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    print(politics.shape)


    tokenized_titles = []
    for title in politics['headline']:
        words = title.lower().replace('"', "").split()
        #words = [word.text for word in tkz(title.replace('"', '').replace('\'', '').lower())]
        words.append('<EOS>')
        tokenized_titles.append(words)


    print("Examples: ", tokenized_titles[0:2])


    # Initialize word counts and a set for unique words
    word_counts = {}
    vocab = set()

    # Conta le parole e raccogli le parole uniche in un unico ciclo
    for title in tokenized_titles:
        for word in title:
            if word != "<EOS>":
                word_counts[word] = word_counts.get(word, 0) + 1
                vocab.add(word)

    vocab = ['<EOS>'] + sorted(vocab) + ['PAD']

    # Create mappings
    word_to_int = {word: idx for idx, word in enumerate(vocab)}
    int_to_word = {idx: word for word, idx  in word_to_int.items()}

    # print the 5 common words
    print("5 most common words:", sorted(word_counts, key=word_counts.get, reverse=True)[:5])
    print("Number of unique words:", len(vocab))


    batch_size = 32
    dataset = Dataset(tokenized_titles, word_to_int)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=lambda b: collate_fn(b, word_to_int["PAD"]),
                            shuffle=True)

    print(dataset[0])

    sentence = "the president wants"

    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
    model = Model(word_to_int, 1024, 150, n_layers=1).to(DEVICE)

    
    print("-= Sentence with sampling strategy =-")
    for i in range(3):
        print_sentence(model, "random", 20)

    print("-= Sentence with greedy strategy =-")
    for i in range(3):
        print_sentence(model, "greedy", 20)


    print("Training LSTM...")
    model, costs, loss_history, perplexity_history = train(model, dataloader, 6, criterion, lr=1e-3,
                                    print_every=1, clip=1)


    criterion_truncated = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
    model_truncated = Model(word_to_int, 2048, 150, n_layers=1).to(DEVICE)

    print("Training LSTM Truncated...")
    model_truncated, costs_truncated, loss_history_truncated, perplexity_history_truncated = train_truncated(model_truncated, dataloader, 6, criterion_truncated, lr=1e-3,
                                    print_every=1, clip=1)


    """
    Plotting the results
    """

    plt.plot(np.array(loss_history), label='Training Loss', color='red', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=1.5, color='blue', linestyle='--', linewidth=1, label='Threshold (1.5)')
    plt.legend()
    plt.savefig('loss_base.png')
    plt.show()


    plt.plot(np.array(perplexity_history), label='Perplexity', color='green', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.title("Perplexity over epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('perplexity_base.png')
    plt.show()


    plt.plot(np.array(loss_history_truncated), label='Training Loss', color='red', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss (Truncated) over epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=1.5, color='blue', linestyle='--', linewidth=1, label='Threshold (1.5)')
    plt.legend()
    plt.savefig('loss_trunc.png')
    plt.show()


    plt.plot(np.array(perplexity_history_truncated), label='Perplexity', color='green', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.title("Perplexity over epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('perplexity_trunc.png')
    plt.show()

    """
    Evaluation of the model after training
    """

    print("-= Evaluation: Sentence with sampling strategy =-")
    for i in range(3):
        print_sentence(model_truncated, "random")

    print("-= Evaluation: Sentence with greedy strategy =-")
    for i in range(3):
        print_sentence(model_truncated, "greedy")

