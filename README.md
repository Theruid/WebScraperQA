# RAG (Retrieval-Augmented Generation) System
<img width="1485" height="474" alt="image" src="https://github.com/user-attachments/assets/4e753c0d-c37c-4cf4-9147-f3e334b26695" />

A powerful RAG system that allows you to query documents using local or cloud-based language models. This system supports both Ollama (local) and Google AI (cloud) models for generating answers based on your indexed documents.

## 🌟 Features

### Core Functionality
- **Document Indexing**: Scrape and index web pages with configurable crawl depth
- **File Upload**: Upload and index PDF, TXT, and Markdown files directly
- **Hybrid Search**: Combines vector similarity (60%) with keyword matching (40%) for better results
- **Conversational Memory**: Chat with your documents with follow-up question support
- **Multiple Model Support**:
  - Local models via Ollama (qwen3:4b, llama3.1:8b, mistral:7b, gemma2:9b, phi3:medium, qwen2.5:7b)
  - Cloud models via Google AI (Gemini 2.5 Flash)

### User Interface
- **Modern Chat Interface**: Ask questions in a natural conversation flow
- **Real-time Scraping Progress**: Live progress bar, ETA, and logs during web scraping
- **Markdown Rendering**: Answers and search results display with proper formatting
- **Dark Mode**: Full dark theme support
- **Glassmorphism UI**: Modern, premium design with smooth animations
- **Toast Notifications**: Elegant feedback for all actions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running (for local models)
- Google AI API key (for Gemini 2.5 Flash)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## 🛠️ Usage

### Adding Sources

**Web Scraping:**
1. Go to "Add Source" → "Web Scraper" tab
2. Enter a URL, optional name, crawl depth, and max pages
3. Click "Start Scraping" and watch real-time progress
4. Content is automatically chunked and indexed

**File Upload:**
1. Go to "Add Source" → "File Upload" tab
2. Drag & drop or click to upload PDF, TXT, or Markdown files
3. Files are automatically processed and indexed

### Searching & Asking Questions

**Semantic Search:**
- Go to the "Search" section
- Enter your query and see results ranked by hybrid score (vector + keyword)

**Chat with Documents:**
- Use the Dashboard chat interface
- Ask questions and get AI-generated answers with source citations
- Ask follow-up questions - the system remembers context
- Click "New Chat" to start a fresh conversation

### Settings

- Choose between Ollama (local) or Google AI (cloud)
- For Google AI, enter your API key
- Test the connection before saving

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key
EMBEDDING_MODEL=google/embeddinggemma-300m
LLM_MODEL=qwen3:4b
```

## 📂 Project Structure

```
rag-system/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # Frontend logic
├── templates/
│   └── index.html        # Main application UI
└── data/
    └── vectors.db        # SQLite database with vector storage
```

## 🤖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scrape` | POST | Scrape URL with SSE progress streaming |
| `/api/upload` | POST | Upload and index a file (PDF, TXT, MD) |
| `/api/search` | POST | Hybrid search (vector + keyword) |
| `/api/answer` | POST | Generate answer with conversational context |
| `/api/stats` | GET | Get document statistics |
| `/api/delete-source` | POST | Remove a source and its chunks |
| `/api/test-model` | POST | Test model connection |

## 📦 Dependencies

- **Flask** - Web framework
- **sentence-transformers** - Text embeddings
- **sqlite-vec** - Vector storage in SQLite
- **ollama** - Local LLM interface
- **google-generativeai** - Gemini API
- **beautifulsoup4** - Web scraping
- **PyMuPDF** - PDF text extraction
- **marked.js** - Markdown rendering (frontend)

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to local LLMs
- [Google AI](https://ai.google.dev/) for the Gemini models
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [SQLite Vector](https://github.com/asg017/sqlite-vector) for efficient vector storage
