# Quick Start Guide

Get started with Persian News Classification in 5 minutes!

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- Internet connection (for downloading models)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd news_classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- beautifulsoup4 (web scraping)
- transformers (BERT model)
- torch (deep learning framework)
- requests (HTTP library)

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `persian_news_bert (2).ipynb` in your browser.

## Quick Examples

### Example 1: Classify Existing Articles

Run this code to classify the included sample articles:

```python
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import re

# Load the model
model_name = "HooshvareLab/bert-fa-base-uncased-clf-persiannews"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Classify a file
with open('content_524.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    result = classifier(text[:512])  # First 512 chars
    print(f"Category: {result[0]['label']}")
    print(f"Confidence: {result[0]['score']:.2%}")
```

Expected output:
```
Category: ورزشی
Confidence: 99.95%
```

### Example 2: Classify Your Own Text

```python
from transformers import pipeline

# Load classifier
classifier = pipeline(
    "text-classification",
    model="HooshvareLab/bert-fa-base-uncased-clf-persiannews"
)

# Your Persian text
persian_text = "تیم فوتبال استقلال امروز با نتیجه دو بر یک پیروز شد"

# Classify
result = classifier(persian_text)

print(f"Category: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2%}")
```

Expected output:
```
Category: ورزشی
Confidence: ~98%
```

### Example 3: Scrape New Articles

```python
import requests
from bs4 import BeautifulSoup

# Scrape Mashreghnews
url = 'https://mashreghnews.ir/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find news links
links = []
for a_tag in soup.find_all('a', href=True):
    if '/news/' in a_tag['href']:
        links.append(a_tag['href'])

print(f"Found {len(links)} news articles")

# Get first article
if links:
    article_url = f"https://mashreghnews.ir{links[0]}"
    article_response = requests.get(article_url)
    article_soup = BeautifulSoup(article_response.text, 'html.parser')

    article_body = article_soup.find('div', {'itemprop': 'articleBody'})
    if article_body:
        print(article_body.get_text()[:200])  # First 200 chars
```

### Example 4: Batch Classification

```python
import os
from transformers import pipeline

# Load classifier
classifier = pipeline(
    "text-classification",
    model="HooshvareLab/bert-fa-base-uncased-clf-persiannews"
)

# Get all text files
files = [f for f in os.listdir() if f.endswith('.txt')]

# Classify all
results = {}
for file_name in files:
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        result = classifier(text[:512])
        results[file_name] = {
            'category': result[0]['label'],
            'confidence': result[0]['score']
        }

# Display results
for file, data in results.items():
    print(f"{file}: {data['category']} ({data['confidence']:.2%})")
```

## Understanding the Output

### Categories

The classifier returns one of these categories:

- **سیاسی**: Political news
- **ورزشی**: Sports news
- **اقتصادی**: Economic news
- **اجتماعی**: Social news
- **بین الملل**: International news
- **علمی فناوری**: Science & Technology news

### Confidence Score

- **> 0.95**: Very high confidence (excellent)
- **0.80 - 0.95**: High confidence (good)
- **0.60 - 0.80**: Moderate confidence (acceptable)
- **< 0.60**: Low confidence (review needed)

## Common Issues & Solutions

### Issue 1: Model Download Fails

**Problem**: Internet connection or Hugging Face access issues

**Solution**:
```python
# Try with cache_dir parameter
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased-clf-persiannews",
    cache_dir="./model_cache"
)
```

### Issue 2: Out of Memory

**Problem**: Not enough RAM for model

**Solution**: Process smaller text chunks
```python
# Limit text length
text = text[:500]  # Process only first 500 chars
result = classifier(text)
```

### Issue 3: Encoding Errors

**Problem**: Persian text not displaying correctly

**Solution**: Always use UTF-8 encoding
```python
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()
```

### Issue 4: Token Length Error

**Problem**: Text exceeds BERT's 512 token limit

**Solution**: Truncate text properly
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(model_name)
tokens = tokenizer.encode(
    text,
    max_length=510,
    truncation=True
)
text_truncated = tokenizer.decode(tokens)
result = classifier(text_truncated)
```

## Performance Tips

### 1. Batch Processing

Process multiple articles together:

```python
texts = [file1_text, file2_text, file3_text]
results = classifier(texts)
```

### 2. GPU Acceleration

If you have a GPU:

```python
import torch

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device
)
```

### 3. Model Caching

Download model once, use offline:

```python
# First run - downloads model
model = BertForSequenceClassification.from_pretrained(
    model_name,
    cache_dir="./models"
)

# Subsequent runs - uses cached model
model = BertForSequenceClassification.from_pretrained(
    "./models/models--HooshvareLab--bert-fa-base-uncased-clf-persiannews/..."
)
```

## Next Steps

1. **Explore the Notebook**: Open `persian_news_bert (2).ipynb` and run all cells
2. **Read Documentation**: Check `README.md` for detailed information
3. **Review Data**: See `DATA_DOCUMENTATION.md` for dataset details
4. **Experiment**: Try classifying your own Persian text
5. **Contribute**: Improve the scraper or add new features

## Useful Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HooshvareLab Model Card](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-persiannews)
- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the documentation
3. Search for similar issues online
4. Open an issue in the repository

## License

See the main README.md for license information.

---

Happy classifying! 🚀
