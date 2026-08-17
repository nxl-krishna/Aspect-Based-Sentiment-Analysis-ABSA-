import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URLS = {
    "Restaurants_Train.xml": "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Restaurants_Train_v2.xml",
    "Laptops_Train.xml": "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Laptop_Train_v2.xml",
    "Restaurants_Test.xml": "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Restaurants_Test_Data_phaseB.xml",
    "Laptops_Test.xml": "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Laptops_Test_Data_phaseB.xml"
}

RAW_DIR = os.path.join("data", "raw")

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename, url in URLS.items():
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            logging.info(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logging.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logging.error(f"Failed to download {filename}: {e}")
        else:
            logging.info(f"File {filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_data()
