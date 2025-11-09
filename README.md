# RAG (Retrieval-Augmented Generation) System
<img width="1280" height="603" alt="image" src="https://github.com/user-attachments/assets/539b98f1-47d4-4b2e-a78a-a1bc0aec5cdb" />

A powerful RAG system that allows you to query documents using local or cloud-based language models. This system supports both Ollama (local) and Google AI (cloud) models for generating answers based on your indexed documents.

## 🌟 Features

- **Document Indexing**: Scrape and index web pages or documents
- **Semantic Search**: Find relevant content using vector similarity
- **Multiple Model Support**:
  - Local models via Ollama (qwen3:4b, llama3.1:8b, mistral:7b, gemma2:9b, phi3:medium, qwen2.5:7b)
  - Cloud models via Google AI (Gemini 2.5 Flash)
- **Web Interface**: Intuitive UI for managing sources and querying documents
- **SQLite Vector Storage**: Efficient storage and retrieval of document embeddings

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

1. **Add Sources**:
   - Click on "Add Source" in the sidebar
   - Enter a URL and optional name
   - Click "Scrape & Add" to index the content

2. **Search Documents**:
   - Use the search bar to find relevant content
   - Results are ranked by semantic similarity

3. **Ask Questions**:
   - Go to the "Ask Question" section
   - Type your question and click "Ask"
   - The system will retrieve relevant context and generate an answer

4. **Manage Settings**:
   - Click on "Settings" in the sidebar
   - Choose between Ollama (local) or Google AI (cloud)
   - For Google AI, enter your API key
   - Test the connection and save your settings

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
GOOGLE_API_KEY=your_google_ai_api_key
EMBEDDING_MODEL=google/embeddinggemma-300m
LLM_MODEL=qwen3:4b
```

### Model Configuration

- **Ollama Models**: Select from available models in the settings
- **Google AI**: Uses Gemini 2.5 Flash (fixed model)

## 📂 Project Structure

```
rag-system/
├── app.py                # Main application entry point
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
│   └── index.html        # Main application UI
└── data/                 # Database and vector storage
    └── vectors.db        # SQLite database for document storage
```

## 🤖 API Endpoints

- `POST /api/scrape` - Add a new URL to the index
- `POST /api/search` - Search indexed documents
- `POST /api/answer` - Get an answer to a question
- `GET /api/stats` - Get document statistics
- `POST /api/delete-source` - Remove a source and its chunks
- `POST /api/test-model` - Test model connection

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to local LLMs
- [Google AI](https://ai.google.dev/) for the Gemini models
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [SQLite Vector](https://github.com/asg017/sqlite-vector) for efficient vector storage
