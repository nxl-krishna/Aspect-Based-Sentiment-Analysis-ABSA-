import re
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("en_core_web_sm not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def clean_text(text: str) -> str:
    """
    Applies basic text cleaning.
    Note: For BIO tagging, we often want to keep the original text to maintain offsets.
    This function is for general cleaning where offset preservation is not strict.
    """
    text = str(text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str):
    """
    Tokenizes the text using spaCy.
    Returns a list of string tokens.
    """
    if nlp is None:
        # Fallback to simple split if spacy isn't loaded
        return text.split()
    doc = nlp(str(text))
    return [token.text for token in doc]

def generate_bio_tags(text: str, aspect: str, from_idx: int, to_idx: int):
    """
    Generates BIO tags for a given sentence and aspect based on character indices.
    Returns a list of tuples: (token, tag)
    """
    if nlp is None:
        raise RuntimeError("spaCy model required for BIO tagging")
        
    doc = nlp(text)
    tags = []
    
    for token in doc:
        # Check if the token's character offsets overlap with the aspect's offsets
        if token.idx >= from_idx and (token.idx + len(token.text)) <= to_idx:
            if token.idx == from_idx:
                tags.append((token.text, "B-ASP"))
            else:
                tags.append((token.text, "I-ASP"))
        else:
            tags.append((token.text, "O"))
            
    return tags

if __name__ == "__main__":
    # Simple test
    sentence = "The battery life is terrible but the display is excellent."
    aspect = "battery life"
    start = sentence.find(aspect)
    end = start + len(aspect)
    
    tags = generate_bio_tags(sentence, aspect, start, end)
    print(tags)
