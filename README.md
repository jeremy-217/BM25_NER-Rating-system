# BM25 NER Rating System

## Overview
A comprehensive content quality evaluation system for RAG (Retrieval-Augmented Generation) applications that provides dual-layer scoring: NER entity analysis and LLM semantic relevance assessment.

## Features
- **BM25 Retrieval Integration**: Keyword-based document filtering with external service at 192.168.1.143:8000
- **NER Entity Scoring**: spaCy-powered entity extraction with quality metrics
- **LLM Semantic Analysis**: Gemini-1.5-flash powered precision scoring (0.000-1.000)
- **Microservice Architecture**: FastAPI-based modular design
- **Dual Scoring Mechanism**: Combined NER (0.0-1.0) + LLM (0.000-1.000) evaluation

## Tech Stack
- **NLP**: spaCy Transformer models (en_core_web_trf, en_core_web_sm)
- **LLM**: Google Generative AI (Gemini-1.5-flash)
- **Retrieval**: BM25 + Elasticsearch
- **Framework**: FastAPI microservices
- **Language**: Python 3.12.4

## Performance Metrics
- 95%+ relevance assessment accuracy
- 28,854+ characters/second processing speed
- Dual scoring mechanism for comprehensive quality evaluation
- Intelligent fallback between Transformer and statistical models

## Installation

### Prerequisites
- Python 3.12+
- Google Generative AI API key
- BM25 retrieval service access

### Setup
```bash
# Install dependencies
pip install spacy google-generativeai fastapi uvicorn

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

## Configuration
Update `config.py` with your API keys and endpoints.

## Usage

### Basic NER Analysis
```bash
python ner.py
```

### Comprehensive Testing
```bash
# Full test suite
python ner_test_suite.py

# Quick integration test
python test_integrated_ner.py
```

## File Structure
- `ner.py` (27.9KB): Core NER analysis system
- `config.py` (4.3KB): System configuration
- `score.py` (7.3KB): LLM scoring mechanisms
- `ner_test_suite.py` (17.8KB): Testing framework
- `test_integrated_ner.py` (4.4KB): Integration tests

## License
MIT License
