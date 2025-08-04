# BM25 NER Rating System

A dual-layer scoring system for RAG content quality evaluation

## Description

This system provides comprehensive content quality assessment for Retrieval-Augmented Generation (RAG) applications through dual evaluation mechanisms. It combines Named Entity Recognition (NER) analysis with LLM-powered semantic relevance scoring to deliver precise content quality metrics for improved RAG performance.

## Key Features

**BM25 Retrieval** - Keyword-based document filtering  
**NER Entity Analysis** - spaCy-powered entity extraction and scoring  
**LLM Semantic Evaluation** - Gemini-1.5-flash precision semantic relevance analysis  
**High Performance** - 28K+ characters/second processing speed  

## Tech Stack

- **Python 3.12** + FastAPI microservices
- **spaCy** (Transformer + statistical models)
- **Google Generative AI**
- **BM25 + Elasticsearch**

## Quick Start

### Requirements
- Python 3.12+
- Google Generative AI API key

### Installation
```bash
# Install packages
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

# Configure API key
# Edit config.py with your API credentials
```

## Usage

```bash
# Basic NER analysis
python ner.py

# Full test suite
python ner_test_suite.py

# Integration test
python test_integrated_ner.py
```

## File Structure

| File | Description |
|------|-------------|
| `ner.py` | Core NER analysis system |
| `score.py` | LLM scoring mechanism |
| `config.py` | System configuration |
| `ner_test_suite.py` | Testing framework |

## License
MIT License
