import pandas as pd
import json
import gradio as gr
import logging
import pickle
from torch.nn import CosineSimilarity
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

if os.path.exists('models_and_functions.pkl'):
    logger.info("Loading models and functions...")
    with open('models_and_functions.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    logger.error("Models and functions not found. Please run this command to save the models and functions.")
    logger.error("python save_models_and_functions.py")
    exit()

text_preprocessor = data['text_preprocessor']
encode_text_bert = data['encode_text_bert']
encode_text_sbert = data['encode_text_sbert']
bert_tokenizer = data['bert_tokenizer']
bert_model = data['bert_model']
sbert_model = data['sbert_model']

df = pd.read_csv('data/data_resume.csv')
df.drop_duplicates(inplace=True)
labels_dict = {label: idx for idx, label in enumerate(df.Category.unique())}
df.Category = df.Category.apply(lambda x: labels_dict[x]).astype(int)

logger.info("Applying text preprocessing...")
df['cleaned_resume'] = text_preprocessor.transform(df['Resume'])
logger.info("Text preprocessing completed.")

def calculate_cosine_similarity(encoded_job, encoded_resumes):
    cosine_sim = CosineSimilarity(dim=1)
    return cosine_sim(encoded_job, encoded_resumes)

# Function to `ma`tch candidates
def match_candidates(job_description, model_choice):
    logger.info("Encoding job description...")
    if model_choice == "BERT":
        encoded_job = encode_text_bert([job_description])
        encoded_resumes = encode_text_bert(df.Resume.tolist())
        logger.info("Job description encoded with BERT.")
        
        logger.info("Calculating similarity scores with BERT...")
        df['similarity_score'] = calculate_cosine_similarity(encoded_job, encoded_resumes)
        logger.info("Similarity scores calculated with BERT.")
    else:
        encoded_job = sbert_model.encode([job_description], convert_to_tensor=True)
        encoded_resumes = sbert_model.encode(df.Resume.tolist(), convert_to_tensor=True)
        logger.info("Job description encoded with SBERT.")
        
        logger.info("Calculating similarity scores with SBERT...")
        df['similarity_score'] = calculate_cosine_similarity(encoded_job, encoded_resumes)
        logger.info("Similarity scores calculated with SBERT.")
    
    print(df.head())
    df_ranked = df.sort_values(by='similarity_score', ascending=False)
    top_candidates = df_ranked.head(5)

    return json.dumps(top_candidates[['similarity_score', 'Resume']].to_dict(orient='records'))

def gradio_interface(job_description, model_choice):
    return match_candidates(job_description, model_choice)

inputs = [gr.Textbox(lines=10, placeholder="Enter job description here..."), 
          gr.Dropdown(choices=["BERT", "SBERT"], value="SBERT", label="Model Choice")]
outputs = gr.JSON(label="Top 5 Candidates")
demo = gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs, title="Job Description Matcher", allow_flagging="never", theme=gr.themes.Soft())

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(server_port=8080, debug=True, share=False)
