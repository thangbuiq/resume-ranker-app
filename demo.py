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
else:
    logger.error("Models and functions not found. Please run this command to save the models and functions.")
    logger.error("python save_models_and_functions.py")
    logger.info("Running the command to save the models and functions...")
    # use subprocess to run the command to save the models and functions
    import subprocess
    subprocess.run(["python", "save_models_and_functions.py"])
    logger.info("Models and functions saved successfully.")

with open('models_and_functions.pkl', 'rb') as f:
    data = pickle.load(f)
        
text_preprocessor = data['text_preprocessor']
encode_text_bert = data['encode_text_bert']
encode_text_sbert = data['encode_text_sbert']
bert_tokenizer = data['bert_tokenizer']
bert_model = data['bert_model']
sbert_model = data['sbert_model']

df = pd.read_csv('data/data_linkedin.csv')
df.drop_duplicates(inplace=True)
labels_dict = {label: idx for idx, label in enumerate(df.category.unique())}
df.category = df.category.apply(lambda x: labels_dict[x]).astype(int)

df['name_candidate'] = df['Name']

# Gộp các cột thông tin thành một cột resume
df['Resume'] = df.apply(lambda row: ' '.join([
    str(row['description']) if pd.notna(row['description']) else '',
    str(row['clean_skills']) if pd.notna(row['clean_skills']) else '',
    str(row['Experience']) if pd.notna(row['Experience']) else '',
]), axis=1)

# Chỉ giữ lại các cột index, name_candidate và resume
df = df[['index', 'name_candidate','category', 'Resume', 'linkedin', 'Experience', 'description']]

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
        encoded_resumes = encode_text_bert(df.cleaned_resume.tolist())
        logger.info("Job description encoded with BERT.")
        
        logger.info("Calculating similarity scores with BERT...")
        df['similarity_score'] = calculate_cosine_similarity(encoded_job, encoded_resumes)
        logger.info("Similarity scores calculated with BERT.")
    else:
        encoded_job = sbert_model.encode([job_description], convert_to_tensor=True)
        encoded_resumes = sbert_model.encode(df.cleaned_resume.tolist(), convert_to_tensor=True)
        logger.info("Job description encoded with SBERT.")
        
        logger.info("Calculating similarity scores with SBERT...")
        df['similarity_score'] = calculate_cosine_similarity(encoded_job, encoded_resumes)
        logger.info("Similarity scores calculated with SBERT.")
    
    print(df.head())
    df_ranked = df.sort_values(by='similarity_score', ascending=False)
    top_candidates = df_ranked.head(5)
    markdown = ""
    
    for idx, row in top_candidates.iterrows():
        markdown += f"### Candidate {idx+1} - {row['name_candidate']} \n"
        markdown += f"- **Similarity Score**: {row['similarity_score']:.4f} \n"
        markdown += f"- **Description**: {row['description']} \n"
        if len(row['Resume']) > 500:
            display_resume = row['Resume'][:500] + "..."
        else:
            display_resume = row['Resume']
        markdown += f"- **Resume**: {display_resume} \n"
        markdown += f"- **Contact via LinkedIn**: {row['linkedin']} \n\n"
        
    return markdown

def gradio_interface(job_description, model_choice):
    return match_candidates(job_description, model_choice)

inputs = [gr.Textbox(lines=10, placeholder="Enter job description here..."), 
          gr.Dropdown(choices=["BERT", "SBERT"], value="SBERT", label="Model Choice")]

# return top-K candidates with their similarity scores, name, resume, and display their profile picture
outputs = [gr.Markdown(label="Top-K Candidates")]
demo = gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs, title="Job Description Matcher", allow_flagging="never", theme=gr.themes.Soft())

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(server_port=8080, debug=True, share=False, server_name="localhost")
