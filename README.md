# ThesisSummarizer
• ThesisSummarizer: Python-based LLM system for academic text summarization
– Utilized Natural Language Processing techniques
– Leveraged pre-trained models from Hugging Face
– Designed system to extract key insights and main ideas from student project theses
1. Python-Based LLM System for Academic Text Summarization
Core Functionality:
The ThesisSummarizer class processes PDF theses to generate summaries and extract key insights using modern NLP techniques.
2. Natural Language Processing Techniques
Text Preprocessing:
Uses nltk for sentence tokenization, stopword removal, and part-of-speech tagging.
Advanced Analysis:
Leverages spacy for:
Named Entity Recognition (identifying entities like organizations, dates, etc.)
Technical Term Extraction (nouns/proper nouns with term frequency analysis)
Key Phrase Extraction:
Uses KeyBERT to identify domain-specific key phrases (1-3 gram ranges).
3. Pre-Trained Models from Hugging Face
Summarization Engine:
Uses the t5-base transformer model via Hugging Face's pipeline API for abstractive summarization.
Chunk Processing:
Splits long texts into manageable chunks (≤512 tokens) to work within transformer model limits.
4. Key Insight Extraction
Structured Output:
Returns:
Condensed Summary (via T5 model)
Key Phrases (via KeyBERT)
Technical Terms (frequency-based)
Named Entities (via spaCy)
PDF Handling:
Includes PyPDF2 integration for direct thesis PDF text extraction.
