# Buildables Capstone Project - Character-level LSTM Text Generator 

This repository contains the Buildables Capstone Project (Month 3) — an implementation of a locally trained character-level LSTM for text generation and analysis. 
The project demonstrates how language modelling works at a basic level — generating text one character at a time by learning from sequences from English text. It simulates the early foundations of large-scale text models (like GPT) in a manageable, interpretable and local form.
The model was trained using PyTorch, using open public-domain data, and was deployed via Streamlit as a simple web interface. 

## Dataset Description 

| Source            | Hugging Face Datasets → monology/pile-uncopyrighted                                  |
|-------------------|--------------------------------------------------------------------------------------|
| Type              | Public-domain English text dataset (Wikipedia, Books3, PubMed, StackExchange, etc.)  |
| Full Size         | ≈ 825 GB (~20 million documents)                                                     |
| Selected Portion  | ≈ 0.01 % subset (~100k documents)                                                    |
| Reason for Choice | Public-domain, diverse linguistic content, safe and suitable for LLM training        |
| Streaming Mode    | Enabled — no local download required                                                 |

**Results:** 

| Metric                   | Value                                                    |
|---------- ---------------|----------------------------------------------------------| 
| Documents processed      | 100,000                                                  |
| Average raw text length  | ≈ 5002 characters                                        |
| Average tokenized length | ≈ 344 tokens                                             |
| Output file size         | ~150 MB (CSV + TXT combined)                             |
| Languages                | English only                                             |
| Tokenizer used           | GPT-2                                                    |
| Output columns           | raw_text, tokenized_text                                 |
| Final files              | pile_uncopyrighted_100k.csv, pile_uncopyrighted_100k.txt |

*A 50-MB trimmed text file was used for local training feasibility: `pile_uncopyrighted_50MB.txt`.*

## Repository Structure 

BuildablesCapstoneProject/
│
├── requirements.txt                → Dependencies for running the app
├── trim_text.py                    → Script used to trim the dataset to ~50MB subset
│
└── Capstone-Project/
    ├── .ipynb_checkpoints/         → Auto-generated Jupyter backups
    ├── char_lstm_model.pth         → Trained PyTorch model weights
    ├── evaluation_outputs.csv      → Generated text outputs from model evaluation
    ├── pile_uncopyrighted_50MB.txt → Trimmed dataset file used for training
    │
    ├── stage2-model-training.ipynb → Trains the LSTM model on the text dataset
    │                                Includes:
    │                                 - Data preprocessing
    │                                 - Model definition
    │                                 - Training loop and loss tracking
    │                                 - Text generation test
    │                                 - Model saving
    │
    ├── stage3-model-evaluation.ipynb → Loads the model, evaluates quality, and
    │                                   generates multiple text samples with
    │                                   varying prompts and temperature values.
    │                                   Outputs stored in `evaluation_outputs.csv`.
    │
    ├── LLM_app.py                   → Streamlit app that loads the trained model
    │                                  and generates text interactively from a
    │                                  user prompt.

## How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/BuildablesCapstoneProject.git
   cd BuildablesCapstoneProject/Capstone-Project
3. Install dependencies:
   pip install -r ../requirements.txt
5. Train the model:
   jupyter notebook stage2-model-training.ipynb
7. Evaluate the model:
   jupyter notebook stage3-model-evaluation.ipynb
9. Run the web app:
    streamlit run LLM_app.py

**Streamlit App:** https://buildablescapstoneproject.streamlit.app/ 

## Application UI
Interface:
A minimal and interactive Streamlit interface with:
- Text prompt input
- Sliders for text length and creativity (temperature)
- Real-time generation results

## Key Features
- Completely local — no external API or cloud dependencies
- Character-level training — model learns from raw text
- Interactive deployment — accessible through Streamlit UI
- Modular structure — easily extendable for further NLP tasks
- Transparent experimentation — all notebooks are readable and editable

## Future Work 
Although the model successfully generates coherent patterns, it can be significantly improved.
Future enhancements:
- Train longer — increase epochs and dataset size for better linguistic fluency.
- Switch to word-level modeling for more semantic meaning.
- Add validation loss tracking and early stopping to prevent overfitting.
- Add text quality evaluation metrics (perplexity, coherence score, etc.).
- Fine-tune on domain-specific data (medical, sports, etc.) for focused applications.
- Upgrade architecture to GRU or Transformer for performance comparison.
- Expand UI to include download/export options for generated text.
- Implement model versioning and automatic reloading through Streamlit.

