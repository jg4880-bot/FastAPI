from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.bigram_model import BigramModel, SpacyEmbedder

app = FastAPI(title="Bigram + spaCy API", version="1.0")

# Sample corpus for the bigram model
corpus = [
    ("The Count of Monte Cristo is a novel written by Alexandre Dumas. "
     "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge."),
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)
embedder = SpacyEmbedder("en_core_web_sm")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.get("/predict/{word}")
def predict_next(word: str):
    return {"word": word, "next_word": bigram_model.predict_next(word)}

@app.get("/sample/{word}")
def sample_next(word: str):
    return {"word": word, "sampled_next": bigram_model.sample_next(word)}

# —— spaCy 向量与相似度 ——
@app.get("/embed/token")
def embed_token(token: str = Query(..., description="single word")):
    vec = embedder.embed_token(token)
    return {"token": token, "dim": len(vec), "first10": vec[:10]}

@app.get("/embed/text")
def embed_text(text: str = Query(..., description="any sentence")):
    vec = embedder.embed_text(text)
    return {"text": text, "dim": len(vec), "first10": vec[:10]}

@app.get("/similarity")
def similarity(w1: str, w2: str):
    return {"w1": w1, "w2": w2, "similarity": embedder.similarity(w1, w2)}
