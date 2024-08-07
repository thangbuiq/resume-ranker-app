import logging
import os
import pickle
import warnings

import gradio as gr
from torch.nn import CosineSimilarity
from utils.database_loading import load_data

warnings.filterwarnings("ignore")

RESUME_PATH = "data/data_linkedin.csv"

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

if os.path.exists("models_and_functions.pkl"):
    logger.info("Loading models and functions...")
else:
    logger.error(
        "Models and functions not found. Please run this command to save the models and functions."
    )
    logger.error("python save_models_and_functions.py")
    logger.info("Running the command to save the models and functions...")
    # use subprocess to run the command to save the models and functions
    import subprocess

    subprocess.run(["python", "save_models_and_functions.py"])
    logger.info("Models and functions saved successfully.")

with open("models_and_functions.pkl", "rb") as f:
    data = pickle.load(f)

text_preprocessor = data["text_preprocessor"]
encode_text_bert = data["encode_text_bert"]
encode_text_sbert = data["encode_text_sbert"]
bert_tokenizer = data["bert_tokenizer"]
bert_model = data["bert_model"]
sbert_model = data["sbert_model"]

df = load_data(RESUME_PATH)

logger.info("Applying text preprocessing...")
df["cleaned_resume"] = text_preprocessor.transform(df["Resume"])
df.drop_duplicates(inplace=True)
logger.info("Text preprocessing completed.")


def calculate_cosine_similarity(encoded_job, encoded_resumes):
    cosine_sim = CosineSimilarity(dim=1)
    return cosine_sim(encoded_job, encoded_resumes)


# Function to `ma`tch candidates
def match_candidates(job_description, top_k=5, threshold=0.3):
    logger.setLevel(logging.INFO)
    logger.info("Encoding job description...")

    encoded_job = sbert_model.encode([job_description], convert_to_tensor=True)
    encoded_resumes = sbert_model.encode(
        df.cleaned_resume.tolist(), convert_to_tensor=True
    )
    logger.info("Job description encoded with SBERT.")

    logger.info("Calculating similarity scores with SBERT...")
    df["similarity_score"] = calculate_cosine_similarity(encoded_job, encoded_resumes)
    logger.info("Similarity scores calculated with SBERT.")

    print(df.head())
    df_ranked = df.sort_values(by="similarity_score", ascending=False)

    df_ranked = df_ranked[df_ranked["similarity_score"] >= threshold]
    top_candidates = df_ranked.head(top_k)

    if len(top_candidates) == 0:
        markdown = "No candidates matching the job description."
    else:
        markdown = f"# Top {top_k} Candidates\n---\n"

        for idx, row in top_candidates.iterrows():
            markdown += f"### Candidate {idx+1} - {row['name_candidate']} \n"
            markdown += f"- **Similarity Score**: {row['similarity_score']:.4f} \n"
            if len(row["Resume"]) > 800:
                display_resume = row["Resume"][:800] + " ...(truncated)"
            else:
                display_resume = row["Resume"]
            markdown += f"- **Resume**: {display_resume} \n"
            markdown += f"- **Contact via LinkedIn**: {row['linkedin']} \n---\n"

    logger.setLevel(logging.ERROR)
    return markdown


def gradio_interface(job_description, top_k):
    return match_candidates(job_description, top_k)


inputs = [
    gr.Textbox(
        lines=10,
        placeholder="Enter job description here...",
        label="Job Description",
    ),
    gr.Slider(
        value=5,
        label="Top-K candidates to display (won't affect the performance)",
        minimum=1,
        maximum=20,
        step=1,
    ),
]

# return top-K candidates with their similarity scores, name, resume, and display their profile picture
outputs = [
    gr.Markdown(
        label="Top-K Candidates",
        show_label=True,
        value="Please submit the job description to get the top-K candidates.",
    )
]
demo = gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=outputs,
    title="RESUME RANKING SYSTEM",
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(server_port=8080, debug=True, share=False, server_name="localhost")
