# Data Documentation

## Overview

This document provides detailed information about the news articles dataset included in this project.

## Dataset Summary

- **Total Articles**: 19 Persian news articles
- **Source**: Mashreghnews.ir (مشرق نیوز)
- **Language**: Persian (Farsi)
- **Format**: Plain text files (.txt)
- **Encoding**: UTF-8

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| سیاسی (Political) | 7 | 36.8% |
| ورزشی (Sports) | 4 | 21.1% |
| اقتصادی (Economic) | 2 | 10.5% |
| اجتماعی (Social) | 3 | 15.8% |
| بین الملل (International) | 2 | 10.5% |
| علمی فناوری (Science & Tech) | 1 | 5.3% |

## File Details

### Political News (سیاسی)

#### content_207.txt
- **Topic**: UK Foreign Ministry announcement
- **Classification Score**: 99.85%
- **Content Summary**: Announcement from UK Foreign Ministry regarding joint actions

#### content_453.txt
- **Topic**: Iranian Foreign Ministry spokesperson
- **Classification Score**: 99.94%
- **Content Summary**: Foreign ministry comments on sanctions

#### content_541.txt
- **Topic**: Terrorism incident in Rask
- **Classification Score**: 99.18%
- **Content Summary**: Report on terrorist attack on police headquarters

#### content_587.txt
- **Topic**: Parliament budget rejection
- **Classification Score**: 95.24%
- **Content Summary**: Parliament speaker's comments after budget rejection

#### content_656.txt
- **Topic**: Nobel Peace Prize
- **Classification Score**: 99.14%
- **Content Summary**: Nobel Peace Prize committee announcement in Oslo

#### content_724.txt
- **Topic**: UK Foreign Ministry (duplicate)
- **Classification Score**: 99.85%
- **Content Summary**: Same content as content_207.txt

#### content_819.txt
- **Topic**: Terrorism analysis
- **Classification Score**: 87.52%
- **Content Summary**: Analysis of terrorist incident

### Sports News (ورزشی)

#### content_462.txt
- **Topic**: Iranian football league
- **Classification Score**: 99.96%
- **Content Summary**: Discussion of conflicts in Premier League

#### content_524.txt
- **Topic**: Kylian Mbappé transfer
- **Classification Score**: 99.95%
- **Content Summary**: Thierry Henry's advice to Mbappé about Real Madrid transfer

#### content_563.txt
- **Topic**: Omid Alishah performance
- **Classification Score**: 99.95%
- **Content Summary**: Alishah's unbeaten derby record

#### content_759.txt
- **Topic**: Mbappé transfer news
- **Classification Score**: 99.95%
- **Content Summary**: Spanish media report on Mbappé's potential transfer

### Economic News (اقتصادی)

#### content_276.txt
- **Topic**: Electric vehicles in Iran
- **Classification Score**: 99.62%
- **Content Summary**: Ministry of Industry interview about electric vehicle plans
- **Notable**: Longest article in dataset (27 lines)

#### content_327.txt
- **Topic**: Economic development
- **Classification Score**: 99.21%
- **Content Summary**: Economic policy discussion

### Social News (اجتماعی)

#### content_102.txt
- **Topic**: Hajj and pilgrimage registration
- **Classification Score**: 99.23%
- **Content Summary**: Detailed announcement about Umrah registration system
- **Notable**: Second longest article (10 lines)

#### content_169.txt
- **Topic**: Gaza ceasefire
- **Classification Score**: 96.45%
- **Content Summary**: Social media post about Gaza ceasefire

### International News (بین الملل)

#### content_596.txt
- **Topic**: Yemen Ansar Allah operations
- **Classification Score**: 99.96%
- **Content Summary**: Interview with Yemeni official about operations

#### content_654.txt
- **Topic**: Gaza conflict
- **Classification Score**: 99.93%
- **Content Summary**: Report on Israeli regime crimes in Gaza

### Science & Technology (علمی فناوری)

#### content_769.txt
- **Topic**: TV network closure
- **Classification Score**: 91.46%
- **Content Summary**: News about "Man O To" network potential closure

## Data Characteristics

### Article Length

- **Shortest**: content_524.txt (7 lines)
- **Longest**: content_276.txt (27 lines)
- **Average**: ~8-10 lines per article

### Content Quality

- All articles are properly formatted Persian text
- Articles contain standard journalistic structure
- Most articles begin with "به گزارش مشرق" (According to Mashregh)
- UTF-8 encoding ensures proper Persian character display

### Classification Accuracy

- **High Confidence (>99%)**: 11 articles (57.9%)
- **Very High Confidence (>95%)**: 16 articles (84.2%)
- **Good Confidence (>85%)**: 19 articles (100%)
- **Average Confidence**: 96.8%

### Naming Convention

Files follow the pattern: `content_{random_number}.txt`
- Random numbers range from 102 to 819
- Numbers are non-sequential
- This reflects the random sampling approach in the scraper

## Data Collection Methodology

1. **Source Selection**: Mashreghnews.ir main page
2. **Link Extraction**: Find all links matching pattern `/news/{y}/{x}`
3. **Random Sampling**: Select random articles from available links
4. **Content Extraction**: Parse `itemprop="articleBody"` div element
5. **File Storage**: Save with random numeric identifier

## Data Usage Notes

### For Training
- This dataset is small (19 articles) and primarily for demonstration
- Not suitable for training from scratch
- Can be used for fine-tuning or evaluation

### For Testing
- Good representation of different news categories
- Suitable for testing Persian NLP pipelines
- Demonstrates real-world news classification scenarios

### For Evaluation
- High classification confidence indicates model quality
- Can benchmark other Persian text classifiers
- Provides baseline for comparison

## Preprocessing Applied

When used in the classification pipeline:

1. **Whitespace Normalization**: Multiple spaces reduced to single space
2. **Newline Handling**: Newlines replaced with spaces
3. **Token Truncation**: Limited to 510 tokens for BERT compatibility
4. **Encoding**: Processed with BERT tokenizer for Persian

## Duplicate Content

**Note**: `content_207.txt` and `content_724.txt` contain identical content about UK Foreign Ministry announcements. This occurred during the random sampling process.

## Ethical Considerations

- Articles are publicly available news content
- Source attribution maintained (Mashreghnews.ir)
- Used for educational and research purposes
- No personal information or sensitive data included

## Future Dataset Expansion

Recommendations for expanding this dataset:

1. **Increase Volume**: Collect 1000+ articles for robust training
2. **Balance Categories**: Ensure equal representation across all categories
3. **Multiple Sources**: Include articles from various Persian news outlets
4. **Temporal Coverage**: Collect articles across different time periods
5. **Metadata**: Add timestamps, authors, and article URLs
6. **Quality Control**: Remove duplicates and validate classifications

## Citation

If using this dataset, please cite:

```
Persian News Classification Dataset
Source: Mashreghnews.ir
Model: HooshvareLab/bert-fa-base-uncased-clf-persiannews
Date: 2024
```

## Data Integrity

All files:
- Use UTF-8 encoding
- Contain valid Persian text
- Are properly formatted
- Include complete article content
- Have been successfully classified by the BERT model

## Contact

For questions about the data or to report issues, please open an issue in the repository.
