# save_models_and_functions.py

import pickle

from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer

from utils.model_loading import encode_text_bert, encode_text_sbert
from utils.text_preprocessing import TextPreprocessor

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create instances of your preprocessor and models
text_preprocessor = TextPreprocessor()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Serialize everything into a pickle file
with open("models_and_functions.pkl", "wb") as f:
    pickle.dump(
        {
            "text_preprocessor": text_preprocessor,
            "encode_text_bert": encode_text_bert,
            "encode_text_sbert": encode_text_sbert,
            "bert_tokenizer": bert_tokenizer,
            "bert_model": bert_model,
            "sbert_model": sbert_model,
        },
        f,
    )

print("Models and functions serialized successfully!")
