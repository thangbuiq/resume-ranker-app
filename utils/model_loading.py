import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text_bert(text):
    input_ids = bert_tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, padding='max_length')
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = bert_model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        mean_hidden_states = last_hidden_states.mean(dim=1)
    return mean_hidden_states.squeeze().numpy()

def encode_text_sbert(text):
    return sbert_model.encode(text)
