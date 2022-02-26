from typing import Optional
import os
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
import numpy as np
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# pls no steal
os.environ["KEY"] = "AIzaSyDfTzEsloZP-gmnI5GCSCu42FfBnHwHK0E"

from fastapi import FastAPI

app = FastAPI()

class FormResponse(BaseModel):
    user_answer: str

# character count, min and max

@app.post("/submit_form/")
def submit_form(form_response: FormResponse):
    if len(form_response.user_answer) > 140:
        return { "error" : "Your text is too long. Please keep it under 140 characters." }
    sentiment = sample_analyze_sentiment(form_response.user_answer)
    return sentiment

def sample_analyze_sentiment(user_answer):
    positive_answers = ['I like the dog.', 'I love the dog.', 'I really like the dog.', 'I really love the dog.']
    negative_answers = ['I dislike the dog.', 'I hate the dog.', 'I really dislike the dog.', 'I really hate the dog.']

    vals = []

    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    for ans in [positive_answers, negative_answers]:
        embeddings = model.encode(ans, convert_to_tensor=True)
        csim = util.cos_sim(embeddings, user_embedding)
        avg = np.average(csim)
        vals.append(avg)


    print(vals)
    print(len(vals))
    return { 'positive': float(vals[0]), 'negative': float(vals[1]) }