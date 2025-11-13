import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import re

# PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(current_dir, "char_lstm_model_extended.pth")
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
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, use_dropout=True):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if use_dropout:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        else:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try extended model first (with dropout)
try:
    model = CharLSTM(vocab_size, use_dropout=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    st.success("✅ Extended model loaded successfully!")
except:
    # Fallback to original model (without dropout)
    try:
        model = CharLSTM(vocab_size, use_dropout=False).to(device)
        original_model_path = os.path.join(current_dir, "char_lstm_model.pth")
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        st.info("ℹ️ Using original model (extended model not found)")
    except:
        st.error("❌ Could not load any model. Please check model files.")
        st.stop()

model.eval()

# TEXT GENERATION FUNCTIONS
def clean_text(text):
    text = text.replace("Ġ", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9.,!?\'"\n ]', '', text)
    text = text.replace(" .", ".").replace(" ,", ",")
    text = ' '.join(text.split())
    return text.strip()

def sample_with_top_k(probs, k=15):
    top_k_idx = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_idx]
    top_k_probs /= top_k_probs.sum()
    return np.random.choice(top_k_idx, p=top_k_probs)

def generate_text(model, start_text="Once upon a time", length=300, temperature=0.4):
    model.eval()
    start_text = start_text.lower().strip()
    
    # Handle characters not in vocabulary
    filtered_chars = [c for c in start_text if c in stoi]
    if not filtered_chars:
        start_text = "the study shows"
    else:
        start_text = ''.join(filtered_chars)
    
    input_seq = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated = list(start_text)

    for i in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
            next_idx = sample_with_top_k(probs, k=15)
            next_char = itos[next_idx]
            
            # Stop at sentence boundaries for better readability
            if next_char == '.' and i > 50:
                generated.append(next_char)
                break
                
            generated.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return clean_text(''.join(generated))

# STREAMLIT UI
st.title("Character-Level LSTM Text Generator")
st.write("Generate creative text sequences using your trained LSTM model!")

# Model info display
st.write(f"**Model Loaded:** {MODEL_PATH}")
st.write(f"**Vocabulary Size:** {vocab_size}")

# Suggest better prompts for the academic model
st.write("**Suggested prompts for best results:**")
st.write("• 'the study shows that'")
st.write("• 'research indicates that'")
st.write("• 'according to the data'")
st.write("• 'the analysis reveals'")

prompt = st.text_input("Enter a starting prompt:", "the study shows that")
length = st.slider("Text Length", min_value=50, max_value=200, value=100, step=25)
temperature = st.slider("Creativity (Temperature)", min_value=0.1, max_value=0.8, value=0.3, step=0.1)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        output = generate_text(model, start_text=prompt, length=length, temperature=temperature)
    st.subheader("Generated Text:")
    st.write(output)
    
    # Save generated text button
    if st.button("Save Generated Text"):
        st.download_button(
            label="Download Text",
            data=output,
            file_name="generated_text.txt",
            mime="text/plain"
        )

st.caption("Character-Level LSTM Text Generation (Buildables Capstone Project)")