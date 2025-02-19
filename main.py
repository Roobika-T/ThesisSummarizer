# Install required packages
!pip install transformers
!pip install nltk
!pip install torch
!pip install spacy
!pip install scikit-learn
!pip install keybert
!pip install sentence-transformers

import nltk
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import PyPDF2
from google.colab import drive
import re
import spacy
import warnings
warnings.filterwarnings("ignore")

class ThesisSummarizer:
    def __init__(self):
        # Initialize NLP components
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        # Download the punkt_tab data package
        nltk.download('punkt_tab') # This line was added to download the missing data package.
        
        # Download spacy model
        !python -m spacy download en_core_web_sm
        
        self.nlp = spacy.load('en_core_web_sm')
        self.key_phrase_model = KeyBERT()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize T5 model for summarization
        self.summarizer = pipeline(
            "summarization",
            model="t5-base",
            device=0 if torch.cuda.is_available() else -1
        )
    def extract_key_phrases(self, text, top_n=10):
        """Extract key phrases using KeyBERT"""
        keywords = self.key_phrase_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=top_n
        )
        return [phrase for phrase, _ in keywords]

    def perform_nlp_analysis(self, text):
        """Perform detailed NLP analysis"""
        doc = self.nlp(text)
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract technical terms
        technical_terms = [token.text for token in doc 
                         if token.pos_ in ['NOUN', 'PROPN'] 
                         and token.text.lower() not in self.stop_words]
        
        # Calculate term frequencies
        term_freq = Counter(technical_terms)
        
        return {
            'entities': entities,
            'technical_terms': dict(term_freq.most_common(10)),
            'key_phrases': self.extract_key_phrases(text)
        }

    def chunk_text(self, text, max_chunk_size=512):
        """Split text into smaller chunks for processing"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def generate_summary(self, text, max_length=150, min_length=50):
        """Generate summary using T5 model"""
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            summary = self.summarizer(chunk,
                                    max_length=max_length,
                                    min_length=min_length,
                                    do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        combined_summary = ' '.join(summaries)
        
        # If combined summary is too long, summarize again
        if len(combined_summary.split()) > max_length:
            final_summary = self.summarizer(combined_summary,
                                          max_length=max_length,
                                          min_length=min_length,
                                          do_sample=False)[0]['summary_text']
            return final_summary
            
        return combined_summary

    def read_pdf(self, file_path):
        """Read and extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "
                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    def process_thesis(self, pdf_path):
        """Process thesis with NLP analysis and summarization"""
        # Read PDF
        thesis_text = self.read_pdf(pdf_path)
        
        # Perform NLP analysis
        nlp_analysis = self.perform_nlp_analysis(thesis_text)
        
        # Generate summary
        summary = self.generate_summary(thesis_text)
        
        return {
            'original_length': len(thesis_text.split()),
            'summary': summary,
            'key_phrases': nlp_analysis['key_phrases'],
            'important_entities': nlp_analysis['entities'],
            'technical_terms': nlp_analysis['technical_terms']
        }


if __name__ == "__main__":
    
    # Initialize summarizer
    summarizer = ThesisSummarizer()
    
    # Process thesis
    pdf_path = '/content/drive/MyDrive/The Impact of Machine Learning on Climate Change Prediction.pdf'
    results = summarizer.process_thesis(pdf_path)
    
    # Print results
    print("\nSummary:")
    print(results['summary'])
    print("\nKey Phrases:")
    print(results['key_phrases'])
    print("\nImportant Technical Terms:")
    print(results['technical_terms'])
    print("\nNamed Entities:")
    print(results['important_entities'])
