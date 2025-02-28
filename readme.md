# CDP Documentation Chatbot - Requirements and Setup



### Option 1: Standard Setup

1. **Clone the repository or create the project structure**

   Create the following directory structure:
   ```
   cdp-chatbot/
   ├── app.py
   ├── scraper.py
   ├── processor.py
   ├── requirements.txt
   ├── static/
   │   ├── style.css
   │   └── script.js
   └── templates/
       └── index.html
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a .env file (Optional for OpenAI integration)**

   If you want to use OpenAI for better answer generation:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

   The application will start on `http://localhost:8000`

5. **First-time setup**

   When the application runs for the first time, it will automatically start scraping the documentation from the CDP websites. This may take some time depending on your internet connection.


## Option 2: Docker Setup

# Create the same project structure as in Option 1.

# Create a Dockerfile using the provided code.

# Build the Docker image:
bashCopydocker build -t cdp-chatbot .

# Run the container:
bashCopydocker run -p 8000:8000 cdp-chatbot

# Access the chatbot at http://localhost:8000

#  Features Implemented

# "How-to" Questions:
 The chatbot understands and responds to user questions about how to perform specific tasks in each CDP.
Extract Information from Documentation: The system scrapes and indexes documentation from all four CDPs, preserving the context and structure for accurate retrieval.
Handle Variations in Questions: The solution uses semantic search and NLP to understand question variations and properly identifies questions unrelated to CDPs.
Bonus Features:

# Cross-CDP Comparisons: 
The system can identify comparison questions and provide structured answers comparing features across platforms.
Advanced "How-to" Questions: The processor handles complex questions by retrieving and combining information from multiple sources.



## Technologies Used

FastAPI: Provides a high-performance web framework for the backend
ChromaDB: Lightweight vector database for semantic search
LangChain: Used for document processing and chunking
OpenAI Embeddings: (Optional) Can be enabled for more accurate embeddings
Beautiful Soup: Used for web scraping of documentation
NLTK: Provides natural language processing capabilities for better text chunking

## Advanced Configuration Options

OpenAI Integration: Set the use_openai parameter to True in processor.py and provide an API key to get higher quality answers (optional).
Custom Scraping: You can adjust max_pages_per_cdp in scraper.py to control how much content is scraped.
Manual Documentation Refresh: Use the "Refresh Documentation" button in the UI to update the knowledge base with the latest documentation.

If you want to run it on desktop please first do this first.




## Requirements
```
fastapi==0.103.1
uvicorn==0.23.2
jinja2==3.1.2
pydantic==2.3.0
requests==2.31.0
beautifulsoup4==4.12.2
chromadb==0.4.15
langchain==0.0.286
openai==0.28.1
nltk==3.8.1
python-dotenv==1.0.0
```


## Usage

1. Open your browser and go to `http://localhost:8000`
2. Enter your CDP-related "how-to" questions in the chat interface
3. The chatbot will search the documentation and provide relevant answers

## Features

- **Documentation Scraping**: Automatically scrapes and indexes documentation from Segment, mParticle, Lytics, and Zeotap
- **Natural Language Understanding**: Processes user questions to extract intent and context
- **Semantic Search**: Uses vector embeddings to find the most relevant documentation
- **Cross-CDP Comparisons**: Handles comparison questions across different CDPs
- **Reference Sources**: Provides links to source documentation for further reading

## Advanced Configuration

### Disable OpenAI Integration

If you don't want to use OpenAI for answer generation, modify the `processor.py` file:

```python
# Change from:
processor = QueryProcessor(use_openai=True)

# To:
processor = QueryProcessor(use_openai=False)
```

### Adjust Scraping Parameters

To modify how much documentation is scraped, edit the `scraper.py` file:

```python
# Increase or decrease the maximum pages to scrape per CDP
self.max_pages_per_cdp = 100  # Change this value
```

### Custom Embedding Models

You can implement custom embedding models by modifying the `embedding_function` in `processor.py`.
