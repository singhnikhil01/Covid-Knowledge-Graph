import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv('question.csv')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Function to compute semantic similarity
def compute_similarity(sentence1, sentence2):

    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    return similarity


def remove_duplicate_questions(df):
    seen_questions = set()
    unique_questions = []

    for index, row in df.iterrows():
        question = row['question']
        is_duplicate = any(compute_similarity(question, seen_question) > 0.8 for seen_question in seen_questions)

        if not is_duplicate:
            seen_questions.add(question)
            unique_questions.append(question)

    unique_df = pd.DataFrame({'question': unique_questions})
    return unique_df

# Remove duplicates
unique_df = remove_duplicate_questions(df)
unique_df.to_csv('sementics.csv')
# print(unique_df)

