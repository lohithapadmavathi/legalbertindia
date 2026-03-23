**#Legal Bert India**
Description

An end-to-end NLP pipeline for analyzing legal documents using OCR, transformer-based models, and rule-based systems. The system extracts text, detects language, performs legal-specific normalization, identifies entities, classifies clauses, maps jurisdiction, retrieves precedents, assesses risk, and generates concise summaries.


**Features**
 OCR-based text extraction from PDFs/images
 Language detection
 Legal text normalization & tokenization
 Named Entity Recognition using fine-tuned BERT
 Legal clause classification
 Jurisdiction mapping
 Precedent retrieval using a database
 ML-based legal risk assessment
 Automated case/document summarization
**Tech Stack**


Language: Python
Libraries: NumPy, PIL, pytesseract
ML Models: BERT (fine-tuned for Legal NER & Clause Classification)
Database: SQLite (precedents.db)
Concepts: NLP, OCR, Information Extraction, ML Pipelines


##**Project Structure**
```text
PROJECT1/
│── legal_bert_clause_classifier/
│── legal_bert_ner_finetuned/
│── legal_ner_frozen/
│── ner_finetuned_model/
│── nerscraper/
│── samples/
│── sentences_cleaned/
│
│── pipeline_core.py        # Main pipeline logic
│── precedent_engine.py     # Precedent retrieval system
│── ner_finetuned.py        # NER model handling
│── risk_data_balancer.py   # Data balancing for risk model
│
│── precedents.db           # Legal precedents database
│── *.json                  # Training & processed datasets
│── *.ipynb                 # Experiment notebooks


**Installation & Setup**
1. Clone the repository
git clone https://github.com/lohithpadmavathi/legalbertindia.git
cd legalbertindia
2. Install dependencies
pip install -r requirements.txt
(If requirements.txt is not present, install manually: numpy, pillow, pytesseract, etc.)


**Pipeline Flow**
OCR Extraction → Extract text from document/image
Language Detection → Identify document language
Normalization → Clean and standardize legal text
Tokenization → Break into structured units
NER (BERT) → Extract legal entities
Clause Classification → Identify clause types
Jurisdiction Mapping → Detect legal region
Precedent Engine → Fetch similar past cases
Risk Assessment → Predict legal risk
Summarization → Generate concise output


**Future Improvements**
Web interface (Streamlit/React)
Better multilingual support
Real-time document processing API
Improved legal knowledge graph integration
