import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text_bert(text):
    with torch.no_grad():
        # Tokenize the input text
        inputs = bert_tokenizer(text, add_special_tokens=True, truncation=True, 
                                max_length=200, padding=True, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Pass the tokenized input to the model
        outputs = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        mean_hidden_states = last_hidden_states.mean(dim=1)
    return mean_hidden_states

def encode_text_sbert(text):
    with torch.no_grad():
        embeddings = sbert_model.encode(text, convert_to_tensor=True)
    return embeddings
