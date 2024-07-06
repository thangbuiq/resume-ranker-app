import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import torch
from utils.text_preprocessing import TextPreprocessor
from utils.model_loading import encode_text_bert, encode_text_sbert

# Load the dataset into a pandas DataFrame
df = pd.read_csv('data/data_resume.csv')

# Convert categories to numerical labels
labels_dict = {label: idx for idx, label in enumerate(df.Category.unique())}
df.Category = df.Category.apply(lambda x: labels_dict[x]).astype(int)

# Apply text preprocessing
text_preprocessor = TextPreprocessor()
df['cleaned_resume'] = text_preprocessor.transform(df['Resume'])

# Set seed for reproducibility
def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(86)

# Calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Function to match candidates
def match_candidates(job_description):
    encoded_job_bert = encode_text_bert(job_description)
    encoded_job_sbert = encode_text_sbert(job_description)
    
    df['similarity_score_bert'] = df['cleaned_resume'].apply(lambda x: calculate_cosine_similarity(encode_text_bert(x), encoded_job_bert))
    df['similarity_score_sbert'] = df['cleaned_resume'].apply(lambda x: calculate_cosine_similarity(encode_text_sbert(x), encoded_job_sbert))
    
    df_ranked_bert = df.sort_values(by='similarity_score_bert', ascending=False)
    df_ranked_sbert = df.sort_values(by='similarity_score_sbert', ascending=False)
    
    top_candidates_bert = df_ranked_bert.head(5)
    top_candidates_sbert = df_ranked_sbert.head(5)
    
    return top_candidates_bert[['Category', 'cleaned_resume', 'similarity_score_bert']], top_candidates_sbert[['Category', 'cleaned_resume', 'similarity_score_sbert']]

# Define Gradio UI
def gradio_interface(job_description):
    top_candidates_bert, top_candidates_sbert = match_candidates(job_description)
    return top_candidates_bert.to_dict('records'), top_candidates_sbert.to_dict('records')

# Define Gradio interface components
inputs = gr.Textbox(lines=10, placeholder="Enter job description here")
outputs = [gr.DataFrame(),gr.DataFrame()]
iface = gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs, title="Job Description Matcher")


if __name__ == '__main__':
    iface.launch(share=False)
