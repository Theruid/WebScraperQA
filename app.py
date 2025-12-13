from flask import Flask, render_template, request, jsonify, Response
import sqlite3
import sqlite_vec
import ollama
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import struct
import os
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import google.generativeai as genai
import fitz  # PyMuPDF for PDF text extraction

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
EMBEDDING_MODEL = 'google/embeddinggemma-300m'
EMBEDDING_DIMS = 256
LLM_MODEL = 'qwen3:4b'
DB_FILE = "rag_vectors.db"
TABLE_NAME = "documents"

# Global variables
embedding_model = None
conn = None

def serialize_f32(vector):
    """Convert float vector to bytes for SQLite storage"""
    return struct.pack("%sf" % len(vector), *vector)

def init_db():
    """Initialize SQLite vector database"""
    global conn
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME} USING vec0(
            text TEXT,
            source TEXT,
            embedding float[{EMBEDDING_DIMS}]
        )
    """)
    conn.commit()

def init_model():
    """Initialize embedding model"""
    global embedding_model
    logging.info(f"Loading {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logging.info("Model loaded successfully!")

def token_based_chunking(text, tokenizer, max_tokens=2048, overlap_tokens=100):
    """Token-based chunking using the embedding model's tokenizer"""
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

def get_domain(url):
    """Extract domain from URL"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def is_same_domain(url1, url2):
    """Check if two URLs are from the same domain"""
    return urlparse(url1).netloc == urlparse(url2).netloc

def extract_links(soup, base_url):
    """Extract all valid links from a page"""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        # Only include http/https links
        if full_url.startswith(('http://', 'https://')):
            # Remove fragments and query params for cleaner URLs
            parsed = urlparse(full_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            links.add(clean_url)
    return links

def scrape_page(url, headers):
    """Scrape a single page and return text content"""
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extract text
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up extra whitespace
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        text_content = '\n'.join(lines)
        
        return text_content, soup
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return None, None

@app.route('/api/scrape', methods=['POST'])
def scrape_url():
    """Scrape URL with real-time progress streaming via SSE"""
    data = request.json
    url = data.get('url')
    source_name = data.get('name', url)
    crawl_depth = data.get('crawl_depth', 0)
    max_pages = data.get('max_pages', 10)
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    def generate():
        import json
        
        def send_event(event_type, data):
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            visited = set()
            to_visit = deque([(url, 0)])
            all_text_content = []
            pages_scraped = 0
            
            base_domain = get_domain(url)
            
            yield send_event('start', {
                'message': f'Starting scrape of {url}',
                'max_pages': max_pages,
                'crawl_depth': crawl_depth
            })
            
            while to_visit and pages_scraped < max_pages:
                current_url, depth = to_visit.popleft()
                
                if current_url in visited:
                    continue
                
                visited.add(current_url)
                
                # Progress update - Scraping phase is 0% to 60%
                scrape_progress = (pages_scraped / max_pages) * 60
                remaining = max_pages - pages_scraped
                est_time = remaining * 1 + len(all_text_content) * 2  # ~1s per page + ~2s per page for indexing
                
                yield send_event('progress', {
                    'current_page': pages_scraped + 1,
                    'max_pages': max_pages,
                    'progress': round(scrape_progress, 1),
                    'estimated_seconds': est_time,
                    'log': f'Scraping: {current_url[:60]}...' if len(current_url) > 60 else f'Scraping: {current_url}',
                    'phase': 'scraping'
                })
                
                text_content, soup = scrape_page(current_url, headers)
                
                if text_content and text_content.strip():
                    all_text_content.append({
                        'url': current_url,
                        'text': text_content
                    })
                    pages_scraped += 1
                    
                    yield send_event('log', {
                        'message': f'✓ Scraped {len(text_content)} chars from page {pages_scraped}'
                    })
                    
                    if depth < crawl_depth and soup:
                        links = extract_links(soup, current_url)
                        new_links = 0
                        for link in links:
                            if is_same_domain(link, base_domain) and link not in visited:
                                to_visit.append((link, depth + 1))
                                new_links += 1
                        if new_links > 0:
                            yield send_event('log', {
                                'message': f'  Found {new_links} sublinks to crawl'
                            })
                
                time.sleep(0.5)
            
            if not all_text_content:
                yield send_event('error', {'message': 'No text content found'})
                return
            
            # Processing phase - 60% to 100%
            yield send_event('log', {'message': 'Processing and indexing content...'})
            
            total_stored = 0
            total_pages = len(all_text_content)
            for i, page_data in enumerate(all_text_content):
                page_url = page_data['url']
                text = page_data['text']
                
                # Indexing phase: 60% to 100%
                index_progress = 60 + ((i + 1) / total_pages) * 40
                remaining_pages = total_pages - i - 1
                est_time = remaining_pages * 2  # ~2s per page for indexing
                
                yield send_event('progress', {
                    'current_page': i + 1,
                    'max_pages': total_pages,
                    'progress': round(index_progress, 1),
                    'estimated_seconds': est_time,
                    'log': f'Indexing page {i+1}/{total_pages}...',
                    'phase': 'indexing'
                })
                
                chunks = token_based_chunking(text, embedding_model.tokenizer, max_tokens=2048, overlap_tokens=100)
                
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
                    if len(embedding) > EMBEDDING_DIMS:
                        embedding = embedding[:EMBEDDING_DIMS]
                    
                    page_source = f"{source_name} ({page_url})"
                    conn.execute(f"""
                        INSERT INTO {TABLE_NAME} (text, source, embedding)
                        VALUES (?, ?, ?)
                    """, (chunk, page_source, serialize_f32(embedding.tolist())))
                    total_stored += 1
            
            conn.commit()
            
            yield send_event('complete', {
                'success': True,
                'message': f'Added {total_stored} chunks from {pages_scraped} page(s)',
                'chunks': total_stored,
                'pages': pages_scraped
            })
            
        except Exception as e:
            logging.error(f"Scrape error: {e}")
            yield send_event('error', {'message': str(e)})
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and index a file (PDF, TXT, MD)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = file.filename
        source_name = request.form.get('name', filename)
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        # Extract text based on file type
        text_content = ''
        
        if file_ext == 'pdf':
            # Use PyMuPDF to extract text from PDF
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                text_content += page.get_text() + '\n'
            doc.close()
        elif file_ext in ['txt', 'md', 'markdown']:
            # Read text files directly
            text_content = file.read().decode('utf-8', errors='ignore')
        else:
            return jsonify({'error': f'Unsupported file type: .{file_ext}. Supported: PDF, TXT, MD'}), 400
        
        if not text_content.strip():
            return jsonify({'error': 'No text content found in file'}), 400
        
        # Chunk and embed
        chunks = token_based_chunking(text_content, embedding_model.tokenizer, max_tokens=2048, overlap_tokens=100)
        
        total_stored = 0
        for chunk in chunks:
            embedding = embedding_model.encode(chunk, normalize_embeddings=True)
            if len(embedding) > EMBEDDING_DIMS:
                embedding = embedding[:EMBEDDING_DIMS]
            
            conn.execute(f"""
                INSERT INTO {TABLE_NAME} (text, source, embedding)
                VALUES (?, ?, ?)
            """, (chunk, source_name, serialize_f32(embedding.tolist())))
            total_stored += 1
        
        conn.commit()
        logging.info(f"Uploaded file '{filename}' with {total_stored} chunks")
        
        return jsonify({
            'success': True,
            'message': f'Added {total_stored} chunks from {filename}',
            'chunks': total_stored,
            'filename': filename
        })
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

def hybrid_search(query, top_k=5, vector_weight=0.6, keyword_weight=0.4):
    """
    Hybrid search combining vector similarity with keyword matching.
    Returns list of (text, source, combined_score) sorted by score descending.
    """
    import re
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    if len(query_embedding) > EMBEDDING_DIMS:
        query_embedding = query_embedding[:EMBEDDING_DIMS]
    
    # Get more candidates from vector search for re-ranking
    candidate_count = max(top_k * 3, 15)
    
    cursor = conn.execute(f"""
        SELECT rowid, text, source, distance
        FROM {TABLE_NAME}
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (serialize_f32(query_embedding.tolist()), candidate_count))
    
    candidates = cursor.fetchall()
    
    if not candidates:
        return []
    
    # Prepare query terms for keyword matching (lowercase, split by non-word chars)
    query_terms = set(re.findall(r'\w+', query.lower()))
    
    scored_results = []
    for row in candidates:
        rowid, text, source, distance = row
        
        # Vector score: convert distance to similarity (lower distance = higher score)
        # Distance is typically 0-2 for normalized embeddings
        vector_score = max(0, 1 - distance)
        
        # Keyword score: what fraction of query terms appear in the text?
        text_lower = text.lower()
        source_lower = source.lower() if source else ""
        matching_terms = sum(1 for term in query_terms if term in text_lower or term in source_lower)
        keyword_score = matching_terms / len(query_terms) if query_terms else 0
        
        # Combined score
        combined_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
        
        scored_results.append({
            'id': rowid,
            'text': text,
            'source': source,
            'distance': distance,
            'vector_score': vector_score,
            'keyword_score': keyword_score,
            'score': combined_score
        })
    
    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_results[:top_k]


@app.route('/api/search', methods=['POST'])
def search():
    """Perform hybrid search (vector + keyword)"""
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = hybrid_search(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'id': r['id'],
                    'text': r['text'],
                    'source': r['source'],
                    'distance': r['distance'],
                    'score': r['score']
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer', methods=['POST'])
def answer():
    """Generate answer using RAG with optional chat history"""
    try:
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', 3)
        settings = data.get('settings', {'provider': 'ollama', 'ollamaModel': 'qwen3:4b'})
        history = data.get('history', [])  # List of {role: 'user'|'assistant', content: '...'}
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Use hybrid search for better context retrieval
        search_results = hybrid_search(question, top_k=top_k)
        contexts = [r['text'] for r in search_results]
        
        if not contexts:
            return jsonify({'error': 'No relevant context found'}), 404
        
        # Build conversation history string
        history_text = ""
        if history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in history[-6:]:  # Limit to last 6 messages (3 Q&A pairs)
                role = "User" if msg.get('role') == 'user' else "Assistant"
                history_text += f"{role}: {msg.get('content', '')}\n"
        
        # Build prompt with context and history
        combined_context = "\n\n".join(contexts)
        prompt = f"""Use the following contexts to answer the question comprehensively.
If you don't know the answer based on the provided contexts, just say that you don't know.
Consider the conversation history when answering follow-up questions.

Contexts:
{combined_context}
{history_text}
Current Question: {question}

Answer:"""
        
        # Generate response based on provider
        if settings.get('provider') == 'google':
            # Use Google AI
            api_key = settings.get('googleApiKey')
            if not api_key:
                return jsonify({'error': 'Google API key not configured'}), 400
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            answer_text = response.text
        else:
            # Use Ollama
            model_name = settings.get('ollamaModel', LLM_MODEL)
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            answer_text = response['message']['content']
        
        return jsonify({
            'success': True,
            'answer': answer_text,
            'contexts': contexts
        })
        
    except Exception as e:
        logging.error(f"Answer error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    try:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cursor.fetchone()[0]
        
        cursor = conn.execute(f"SELECT DISTINCT source FROM {TABLE_NAME}")
        sources = [row[0] for row in cursor.fetchall()]
        
        return jsonify({
            'success': True,
            'total_chunks': count,
            'sources': sources,
            'source_count': len(sources)
        })
    except Exception as e:
        logging.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-source', methods=['POST'])
def delete_source():
    """Delete all chunks from a specific source"""
    try:
        data = request.json
        source_name = data.get('source')
        
        if not source_name:
            return jsonify({'error': 'Source name is required'}), 400
        
        # Delete all chunks from this source
        cursor = conn.execute(f"DELETE FROM {TABLE_NAME} WHERE source = ?", (source_name,))
        conn.commit()
        deleted_count = cursor.rowcount
        
        if deleted_count == 0:
            return jsonify({'error': 'Source not found'}), 404
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} chunks from {source_name}',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        logging.error(f"Delete error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-model', methods=['POST'])
def test_model():
    """Test model connection"""
    try:
        settings = request.json
        
        if settings.get('provider') == 'google':
            # Test Google AI
            api_key = settings.get('googleApiKey')
            if not api_key:
                return jsonify({'error': 'API key is required'}), 400
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')  # Fixed to Gemini 2.5 Flash
            response = model.generate_content("Say 'Hello'")
            
            return jsonify({'success': True, 'message': 'Google AI connection successful'})
        else:
            # Test Ollama
            model_name = settings.get('ollamaModel', 'qwen3:4b')
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say hello'}],
                stream=False
            )
            
            return jsonify({'success': True, 'message': 'Ollama connection successful'})
        
    except Exception as e:
        logging.error(f"Test model error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    init_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
