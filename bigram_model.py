import random
import spacy
from typing import Dict, List, Optional

class BigramModel:
    def __init__(self, corpus: List[str]):
        self.model: Dict[str, Dict[str, int]] = {}
        for line in corpus:
            words = line.strip().split()
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.model.setdefault(w1, {})
                self.model[w1][w2] = self.model[w1].get(w2, 0) + 1

    def predict_next(self, word: str) -> Optional[str]:
        if word not in self.model:
            return None
        return max(self.model[word], key=self.model[word].get)

    def sample_next(self, word: str) -> Optional[str]:
        if word not in self.model:
            return None
        choices, weights = zip(*self.model[word].items())
        return random.choices(choices, weights=weights, k=1)[0]

    def generate_text(self, start_word: str, length: int = 5) -> str:

         words = [start_word]
        cur = start_word
        for _ in range(length - 1):
            nxt = self.predict_next(cur)
            if not nxt:
                break
            words.append(nxt)
            cur = nxt
        return " ".join(words)

def sample_word(word_probs: Dict[str, float]) -> str:
    words, probs = zip(*word_probs.items())
    return random.choices(words, weights=probs, k=1)[0]

class SpacyEmbedder:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def embed_token(self, token: str) -> List[float]:
        doc = self.nlp(token)
        return doc[0].vector.tolist()

    def embed_text(self, text: str) -> List[float]:
        doc = self.nlp(text)
        return doc.vector.tolist()

    def similarity(self, w1: str, w2: str) -> float:
        d1, d2 = self.nlp(w1), self.nlp(w2)
        return float(d1.similarity(d2))