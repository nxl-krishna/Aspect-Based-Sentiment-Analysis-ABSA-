import os
import pandas as pd
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def parse_xml_to_df(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'xml')
    sentences = soup.find_all('sentence')
    
    data = []
    
    for sentence in sentences:
        text = sentence.find('text').text
        aspect_terms_tag = sentence.find('aspectTerms')
        
        if aspect_terms_tag:
            for aspect_term in aspect_terms_tag.find_all('aspectTerm'):
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')
                from_idx = aspect_term.get('from')
                to_idx = aspect_term.get('to')
                
                # Exclude conflict polarities as they represent mixed sentiment on the exact same aspect
                if polarity != 'conflict':
                    data.append({
                        'text': text,
                        'aspect': term,
                        'sentiment': polarity,
                        'from': int(from_idx),
                        'to': int(to_idx)
                    })
    
    return pd.DataFrame(data)

def process_all_files():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".xml"):
            filepath = os.path.join(RAW_DIR, filename)
            logging.info(f"Processing {filepath}...")
            df = parse_xml_to_df(filepath)
            
            out_filename = filename.replace('.xml', '.csv')
            out_filepath = os.path.join(PROCESSED_DIR, out_filename)
            df.to_csv(out_filepath, index=False, encoding='utf-8')
            logging.info(f"Saved to {out_filepath}. Shape: {df.shape}")

if __name__ == "__main__":
    process_all_files()
