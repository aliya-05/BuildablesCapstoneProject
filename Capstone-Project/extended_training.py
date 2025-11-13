# Extended training to get loss below 0.8
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# Same setup as before
BASE_DIR = "."
TEXT_FILE = os.path.join(BASE_DIR, "pile_uncopyrighted_50MB.txt")
MODEL_PATH = os.path.join(BASE_DIR, "char_lstm_model_improved.pth")
EXTENDED_MODEL_PATH = os.path.join(BASE_DIR, "char_lstm_model_extended.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load and clean dataset
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

def clean_training_data(text):
    text = text.replace("Ġ", " ")
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

text = clean_training_data(text)
text = text[:1500000]  # Use 1.5M for faster training
print(f"Loaded text length: {len(text):,} characters")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

class TextDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.seq_len+1]
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)
        return input_seq, target_seq

encoded = encode(text)
dataset = TextDataset(encoded, seq_len=100)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # Increased batch size

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Load existing model if available, otherwise create new
model = CharLSTM(vocab_size=vocab_size).to(device)

if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Continuing training from existing model")
else:
    print("Starting fresh training")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)  # Slightly lower LR

# EXTENDED TRAINING - Continue until loss < 0.8
EPOCHS = 30  # More epochs
target_loss = 0.8

print("Starting extended training...")
print(f"Target loss: {target_loss}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output, _ = model(inputs)
        loss = criterion(output.transpose(1, 2), targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
    
    # Save model every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), EXTENDED_MODEL_PATH)
        print(f"Model saved at epoch {epoch+1}")
    
    # Stop if target loss reached
    if avg_loss < target_loss:
        print(f"Target loss {target_loss} reached! Stopping training.")
        break

# Final save
torch.save(model.state_dict(), EXTENDED_MODEL_PATH)
print(f"Final model saved at {EXTENDED_MODEL_PATH}")

# Test generation with improved model
def sample_with_top_k(probs, k=30):  # More focused sampling
    top_k_idx = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_idx]
    top_k_probs /= top_k_probs.sum()
    return np.random.choice(top_k_idx, p=top_k_probs)

def generate_text(model, start_text="hello", length=200, temperature=0.4):  # Lower temperature
    model.eval()
    start_text = start_text.lower()
    input_seq = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated = list(start_text)

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().ravel()
            next_idx = sample_with_top_k(probs, k=30)
            next_char = itos[next_idx]
            generated.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return ''.join(generated)

print("\n=== TESTING EXTENDED MODEL ===")
test_prompts = [
    "in this research we",
    "the study shows that",
    "according to the analysis"
]

for prompt in test_prompts:
    result = generate_text(model, start_text=prompt, length=150)
    print(f"Input: '{prompt}'")
    print(f"Output: {result}")
    print("-" * 50)