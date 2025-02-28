# processor.py
import os
import glob
import re
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
import logging
from scraper import DocumentationScraper
from langchain.embeddings import OpenAIEmbeddings
import openai
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QueryProcessor")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QueryProcessor:
    def __init__(self, use_openai=False):
        self.documentation_dir = "documentation"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.use_openai = use_openai
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        
        # Set up embeddings function
        if self.use_openai:
            # This assumes you have OPENAI_API_KEY in your environment variables
            self.openai_embeddings = OpenAIEmbeddings()
            self.embedding_function = lambda texts: [
                self.openai_embeddings.embed_query(text) for text in texts
            ]
        else:
            # Use a local embedding function from ChromaDB
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create collection
        try:
            self.collection = self.client.get_collection("cdp_documentation")
            logger.info("Using existing ChromaDB collection")
        except:
            self.collection = self.client.create_collection(
                "cdp_documentation",
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            logger.info("Created new ChromaDB collection")
            
            # Load documentation if collection is new
            self.load_documentation()
        
    def chunk_text(self, text: str, cdp: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for better retrieval"""
        # Extract URL from the text
        url_match = re.search(r"URL: (.*)\n", text)
        url = url_match.group(1) if url_match else "Unknown URL"
        
        # Remove the URL line from the text
        if url_match:
            text = text[url_match.end():]
        
        # Use NLTK to split into sentences and then combine into chunks
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "cdp": cdp,
                        "filename": filename,
                        "url": url
                    })
                
                # Start a new chunk with overlap
                overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_point:] + " " + sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "cdp": cdp,
                "filename": filename,
                "url": url
            })
        
        return chunks
                
    def load_documentation(self):
        """Load documentation from files into ChromaDB"""
        logger.info("Loading documentation into ChromaDB")
        
        # Check if documentation exists, if not, run the scraper
        if not os.path.exists(self.documentation_dir) or not os.listdir(self.documentation_dir):
            logger.info("Documentation not found. Running scraper...")
            scraper = DocumentationScraper()
            scraper.scrape_all_parallel()
        
        # Process each documentation file
        all_chunks = []
        ids = []
        metadata = []
        
        for cdp in os.listdir(self.documentation_dir):
            cdp_dir = os.path.join(self.documentation_dir, cdp)
            if not os.path.isdir(cdp_dir):
                continue
                
            for file_path in glob.glob(os.path.join(cdp_dir, "*.txt")):
                filename = os.path.basename(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Chunk the content
                chunks = self.chunk_text(content, cdp, filename)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{cdp}_{filename}_{i}"
                    all_chunks.append(chunk["text"])
                    ids.append(chunk_id)
                    metadata.append({
                        "cdp": chunk["cdp"],
                        "filename": chunk["filename"],
                        "url": chunk["url"]
                    })
        
        # Add to ChromaDB in batches to prevent memory issues
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:end_idx],
                ids=ids[i:end_idx],
                metadatas=metadata[i:end_idx]
            )
            logger.info(f"Added batch {i//batch_size + 1} to ChromaDB")
        
        logger.info(f"Loaded {len(all_chunks)} document chunks into ChromaDB")
    
    def refresh_documentation(self):
        """Refresh documentation by scraping again and reloading"""
        logger.info("Refreshing documentation")
        
        # Clear existing collection
        self.client.delete_collection("cdp_documentation")
        self.collection = self.client.create_collection(
            "cdp_documentation",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        # Run scraper with refresh flag
        scraper = DocumentationScraper()
        scraper.scrape_all_parallel(refresh=True)
        
        # Reload documentation
        self.load_documentation()
    
    def process_query(self, query: str) -> str:
        """Process a user query and return an answer"""
        # Check if query is irrelevant to CDPs
        cdp_terms = ["segment", "mparticle", "lytics", "zeotap", "cdp", "customer data platform"]
        query_lower = query.lower()
        
        # Check if the query is about CDPs
        if not any(term in query_lower for term in cdp_terms):
            # Check if it's a how-to question
            how_to_patterns = [
                r"how (do|can|to|should) (i|we|you)",
                r"what is the (process|way|method) (for|to)",
                r"steps (to|for)",
            ]
            
            if not any(re.search(pattern, query_lower) for pattern in how_to_patterns):
                return "I'm a CDP documentation assistant. I can help with questions about Segment, mParticle, Lytics, and Zeotap. Please ask a question related to these Customer Data Platforms."
        
        # Get results from ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # If no results, return a default message
        if not documents:
            return "I couldn't find specific information about that in the CDP documentation. Could you rephrase your question or be more specific about which CDP feature you're asking about?"
        
        # Handle cross-CDP comparison questions
        if any(cdp in query_lower for cdp in ["segment", "mparticle", "lytics", "zeotap"]) and \
           any(word in query_lower for word in ["compare", "comparison", "difference", "versus", "vs"]):
            return self._generate_comparison_answer(query, documents, metadatas)
        
        # Generate answer using the top results
        answer = self._generate_answer(query, documents, metadatas)
        return answer
    
    def _generate_answer(self, query, documents, metadatas):
        """Generate an answer based on retrieved documents"""
        # Format sources for reference
        sources = []
        for i, metadata in enumerate(metadatas):
            if metadata["url"] not in sources:
                sources.append(metadata["url"])
        
        # Simple answer generation
        relevant_cdp = self._identify_relevant_cdp(query)
        
        # Combine documents to form an answer
        combined_text = "\n\n".join(documents)
        
        # Extract the most relevant sentences for the answer
        sentences = sent_tokenize(combined_text)
        
        # If using OpenAI, we can use the API to generate a better answer
        if self.use_openai and "OPENAI_API_KEY" in os.environ:
            try:
                # Format context with retrieved information
                context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents[:3])])
                
                # Use OpenAI for answer generation
                prompt = f"""
                You are a helpful assistant that answers questions about Customer Data Platforms (CDPs).
                
                Answer the following question based on this context from the CDP documentation:
                
                {context}
                
                Question: {query}
                
                Provide a clear, step-by-step response. If the information is not in the context, say so politely.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful CDP documentation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                
                # Add source references
                answer += "\n\nSources:"
                for i, url in enumerate(sources[:3]):
                    answer += f"\n{i+1}. {url}"
                
                return answer
                
            except Exception as e:
                logger.error(f"Error using OpenAI API: {e}")
                # Fall back to basic answer generation
        
        # Basic answer generation without OpenAI
        answer_sentences = []
        
        # Include the most relevant parts of the documentation
        keyword_matches = self._extract_keywords(query)
        
        for sentence in sentences:
            # Check if the sentence contains keywords from the query
            if any(keyword.lower() in sentence.lower() for keyword in keyword_matches):
                answer_sentences.append(sentence)
        
        # If we don't have enough sentences, add more from the top documents
        if len(answer_sentences) < 5:
            for sentence in sentences:
                if sentence not in answer_sentences:
                    answer_sentences.append(sentence)
                if len(answer_sentences) >= 10:
                    break
        
        # Build the answer
        if relevant_cdp:
            answer = f"Here's how to {query.lower().replace('how do i', '').replace('how can i', '').replace('how to', '')} in {relevant_cdp.capitalize()}:\n\n"
        else:
            answer = f"Here's information about your question:\n\n"
        
        answer += " ".join(answer_sentences[:10])
        
        # Add source references
        answer += "\n\nSources:"
        for i, url in enumerate(sources[:3]):
            answer += f"\n{i+1}. {url}"
        
        return answer
    
    def _generate_comparison_answer(self, query, documents, metadatas):
        """Generate a comparison answer for cross-CDP questions"""
        # Extract CDPs being compared
        cdp_mentions = []
        for cdp in ["segment", "mparticle", "lytics", "zeotap"]:
            if cdp in query.lower():
                cdp_mentions.append(cdp)
        
        # Group documents by CDP
        cdp_documents = {}
        for i, metadata in enumerate(metadatas):
            cdp = metadata["cdp"]
            if cdp not in cdp_documents:
                cdp_documents[cdp] = []
            cdp_documents[cdp].append(documents[i])
        
        # If we have documents for all mentioned CDPs, create a comparison
        answer = f"Here's a comparison of {', '.join(cdp_mentions)}:\n\n"
        
        for cdp in cdp_mentions:
            if cdp in cdp_documents:
                answer += f"## {cdp.capitalize()}\n"
                combined_text = "\n".join(cdp_documents[cdp][:2])
                # Extract a few key sentences
                sentences = sent_tokenize(combined_text)[:5]
                answer += " ".join(sentences) + "\n\n"
            else:
                answer += f"## {cdp.capitalize()}\n"
                answer += f"I couldn't find specific information about {cdp} related to your query.\n\n"
        
        # Add source references
        sources = []
        for metadata in metadatas:
            if metadata["url"] not in sources:
                sources.append(metadata["url"])
        
        answer += "Sources:\n"
        for i, url in enumerate(sources[:4]):
            answer += f"{i+1}. {url}\n"
        
        return answer
    
    def _identify_relevant_cdp(self, query):
        """Identify which CDP the query is about"""
        query_lower = query.lower()
        for cdp in ["segment", "mparticle", "lytics", "zeotap"]:
            if cdp in query_lower:
                return cdp
        return None
    
    def _extract_keywords(self, query):
        """Extract keywords from the query"""
        # Remove common words and keep important terms
        query = query.lower()
        query = re.sub(r'how (do|can|to|should) (i|we|you) ', '', query)
        query = re.sub(r'what is the (process|way|method) (for|to) ', '', query)
        
        # Split into words and filter out common stopwords
        stopwords = ['a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'is', 'are', 'was', 'were']
        words = query.split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords


if __name__ == "__main__":
    processor = QueryProcessor(use_openai=False)
    test_query = "How do I set up a new source in Segment?"
    answer = processor.process_query(test_query)
    print(answer)