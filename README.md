# Persian News Classification with BERT

A machine learning project for automated classification of Persian (Farsi) news articles using BERT transformer model.

## Overview

This project demonstrates web scraping, natural language processing, and text classification for Persian news articles. It uses a pre-trained BERT model specifically fine-tuned for Persian language to classify news articles into different categories.

## Features

- **Web Scraping**: Automated extraction of Persian news articles from Mashreghnews website
- **Text Classification**: Multi-category classification using BERT model
- **Persian Language Support**: Specialized model for Farsi text processing
- **Real-time Processing**: Ability to process and classify news articles on-demand

## News Categories

The model classifies Persian news into the following categories:

- **سیاسی (Political)**: Political news and government affairs
- **ورزشی (Sports)**: Sports news and events
- **اقتصادی (Economic)**: Economic and business news
- **اجتماعی (Social)**: Social and cultural news
- **بین الملل (International)**: International news
- **علمی فناوری (Science & Technology)**: Science and technology news

## Project Structure

```
news_classification/
├── persian_news_bert (2).ipynb    # Main Jupyter notebook
├── content_*.txt                   # Sample news articles (19 files)
├── bert_classification_fa.mp4      # Demo video
└── README.md                       # Project documentation
```

## Requirements

### Python Libraries

```python
beautifulsoup4  # Web scraping
bs4            # BeautifulSoup wrapper
requests       # HTTP requests
transformers   # Hugging Face transformers (BERT)
torch          # PyTorch deep learning framework
torchvision    # PyTorch vision utilities
torchaudio     # PyTorch audio utilities
```

### Installation

```bash
pip install bs4
pip install transformers torch torchvision torchaudio
```

## Usage

### 1. Web Scraping

The notebook includes code to scrape news articles from Mashreghnews:

```python
import requests
from bs4 import BeautifulSoup
import random

url = 'https://mashreghnews.ir/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract links matching pattern: /news/{y}/{x}
links = []
for a_tag in soup.find_all('a', href=True):
    href = a_tag['href']
    if '/news/' in href:
        links.append(href)

# Select random articles and save to files
selected_links = random.sample(links, 5)
```

### 2. Text Classification

Load the pre-trained BERT model and classify articles:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load model
model_name = "HooshvareLab/bert-fa-base-uncased-clf-persiannews"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Classify text
result = classifier(text)
print(f"Label: {result[0]['label']} | Score: {result[0]['score']:.4f}")
```

## Model Details

### BERT Model
- **Model Name**: HooshvareLab/bert-fa-base-uncased-clf-persiannews
- **Source**: Hugging Face Model Hub
- **Language**: Persian (Farsi)
- **Task**: Multi-class text classification
- **Max Sequence Length**: 512 tokens

### Text Preprocessing

The project includes text preprocessing to handle Persian text:

```python
def preprocess_text(text):
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Replace newlines with space
    text = text.replace('\n', ' ')
    return text
```

### Token Limitation

Due to BERT's 512 token limitation, the code processes only the first 510 tokens:

```python
text_for_classification = tokenizer.encode(
    text,
    max_length=510,
    truncation=True,
    return_tensors='pt'
)
```

## Sample Results

The project includes 19 pre-classified news articles demonstrating the model's accuracy:

| File | Category | Confidence |
|------|----------|------------|
| content_524.txt | ورزشی (Sports) | 99.95% |
| content_207.txt | سیاسی (Political) | 99.85% |
| content_276.txt | اقتصادی (Economic) | 99.62% |
| content_596.txt | بین الملل (International) | 99.96% |
| content_102.txt | اجتماعی (Social) | 99.23% |

## Workflow

1. **Scraping**: Extract news articles from Mashreghnews website
2. **Storage**: Save articles as individual text files
3. **Preprocessing**: Clean and prepare text for classification
4. **Tokenization**: Convert text to BERT-compatible tokens
5. **Classification**: Process through BERT model
6. **Output**: Display category label and confidence score

## Technical Notes

- Articles are saved with random numeric identifiers (e.g., content_524.txt)
- The scraper targets articles with `itemprop="articleBody"` attribute
- Text preprocessing handles multiple spaces and newlines
- Token truncation ensures compatibility with BERT's input constraints
- Classification results include both label and confidence score

## Demo

A video demonstration (`bert_classification_fa.mp4`) is included showing the classification process in action.

## File Descriptions

### Code Files
- `persian_news_bert (2).ipynb`: Main notebook containing all scraping and classification logic

### Data Files
The repository includes 19 sample Persian news articles:
- `content_102.txt`: Social news about Hajj and pilgrimage registration
- `content_169.txt`: Social news about Gaza ceasefire
- `content_207.txt`: Political news about UK foreign ministry
- `content_276.txt`: Economic news about electric vehicles
- `content_327.txt`: Economic news article
- `content_339.txt`: Political news about Nobel Foundation
- `content_453.txt`: Political news from foreign ministry
- `content_462.txt`: Sports news about football league
- `content_524.txt`: Sports news about Kylian Mbappé
- `content_541.txt`: Political news about terrorism incident
- `content_563.txt`: Sports news about Omid Alishah
- `content_587.txt`: Political news about parliament
- `content_596.txt`: International news about Yemen
- `content_654.txt`: International news about Gaza
- `content_656.txt`: Political news about Nobel Peace Prize
- `content_724.txt`: Political news (duplicate content with 207)
- `content_759.txt`: Sports news about Mbappé transfer
- `content_769.txt`: Science & Technology news about TV networks
- `content_819.txt`: Political news about terrorism

## Future Improvements

Potential enhancements for this project:

1. **Database Integration**: Store articles in a database instead of text files
2. **Real-time Classification**: Build a web API for live classification
3. **Multi-source Scraping**: Extend to other Persian news websites
4. **Fine-tuning**: Further train the model on domain-specific data
5. **Sentiment Analysis**: Add sentiment classification alongside category detection
6. **Article Summarization**: Implement automatic text summarization
7. **Batch Processing**: Optimize for processing large volumes of articles

## License

This project uses the HooshvareLab BERT model. Please refer to the model's license on Hugging Face for usage terms.

## Acknowledgments

- **HooshvareLab**: For providing the pre-trained Persian BERT model
- **Mashreghnews**: Source of news articles for demonstration
- **Hugging Face**: Transformers library and model hosting

## Contact

For questions or contributions, please open an issue in the repository.

---

**Note**: This project is for educational and research purposes. Ensure compliance with website terms of service when scraping content.
