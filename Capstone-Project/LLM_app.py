import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(current_dir, "char_lstm_model.pth")
TEXT_FILE = os.path.join(current_dir, "pile_uncopyrighted_50MB.txt")

# LOAD DATASET
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

itos = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)

def encode(s): 
    return [stoi.get(c, 0) for c in s]

def decode(l): 
    return ''.join([itos[i] for i in l if i < len(itos)])

# MODEL DEFINITION
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# TEXT GENERATION FUNCTIONS
def clean_text(text):
    text = text.replace("Ġ", " ")
    text = text.replace(" .", ".").replace(" ,", ",")
    text = ' '.join(text.split())
    return text.strip()

def generate_text(model, start_text="Once upon a time", length=300, temperature=0.7):
    model.eval()
    input_seq = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated = list(start_text)

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = itos[next_idx]
            generated.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return clean_text(''.join(generated))

# STREAMLIT UI
st.title("Character-Level LSTM Text Generator")
st.write("Generate creative text sequences using your trained LSTM model!")

prompt = st.text_input("Enter a starting prompt:", "Once upon a time")
length = st.slider("Text Length", min_value=100, max_value=1000, value=300, step=50)
temperature = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.2, value=0.7, step=0.1)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        output = generate_text(model, start_text=prompt, length=length, temperature=temperature)
    st.subheader("Generated Text:")
    st.write(output)

st.caption("Character-Level LSTM Text Generation (Buildables Capstone Project)")