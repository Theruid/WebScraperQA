/**
 * Main Application Logic
 * Handles UI interactions, API calls, and State management.
 */

// --- State & Config ---
const CONFIG = {
    endpoints: {
        stats: '/api/stats',
        scrape: '/api/scrape',
        answer: '/api/answer',
        search: '/api/search',
        testModel: '/api/test-model',
        deleteSource: '/api/delete-source',
        upload: '/api/upload'
    }
};

// --- Toast Notification System ---
const Toast = {
    container: null,

    init() {
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        document.body.appendChild(this.container);
    },

    show(message, type = 'info') {
        if (!this.container) this.init();

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        // Icon based on type
        let icon = '';
        if (type === 'success') icon = '<svg class="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>';
        else if (type === 'error') icon = '<svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>';
        else icon = '<svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';

        toast.innerHTML = `${icon}<span class="text-sm font-medium">${message}</span>`;

        this.container.appendChild(toast);

        // Remove after animation (3s total)
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
};

// --- Theme Manager ---
const ThemeManager = {
    init() {
        const theme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        this.updateIcon(theme);
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        this.updateIcon(next);
    },

    updateIcon(theme) {
        const btn = document.getElementById('themeToggleBtn');
        if (!btn) return;
        if (theme === 'dark') {
            btn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>'; // Sun icon
        } else {
            btn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/></svg>'; // Moon icon
        }
    }
};

// --- Navigation ---
function initNavigation() {
    window.showSection = (sectionId) => {
        // Hide all sections
        document.querySelectorAll('.section').forEach(s => {
            s.style.display = 'none';
            s.classList.remove('active'); // Remove animation class to reset it
        });

        // Deactivate nav items
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

        // Show target section with a slight delay to allow re-triggering animation (optional, but robust)
        const target = document.getElementById(sectionId);
        target.style.display = 'block';
        // Force reflow
        void target.offsetWidth;
        target.classList.add('active'); // Applies fade-in animation defined in CSS

        // Activate nav item
        const navBtn = document.querySelector(`[data-section="${sectionId}"]`);
        if (navBtn) navBtn.classList.add('active');
    };
}

// --- API Interactions ---

async function loadStats() {
    try {
        const res = await fetch(CONFIG.endpoints.stats);
        const data = await res.json();
        if (data.success) {
            document.getElementById('totalChunks').textContent = data.total_chunks;
            document.getElementById('sourceCount').textContent = data.source_count;

            const sourcesList = document.getElementById('sourcesList');
            if (data.sources.length > 0) {
                sourcesList.innerHTML = data.sources.map(s =>
                    `<div class="flex items-center justify-between px-4 py-3 bg-white/50 border border-gray-200/50 rounded-lg text-sm group hover:bg-white/80 transition shadow-sm">
                        <span class="font-medium text-gray-700 dark:text-gray-200">${s}</span>
                        <button onclick="deleteSource('${s.replace(/'/g, "\\'")}')"
                                class="text-red-500 hover:text-red-700 opacity-0 group-hover:opacity-100 transition p-1 rounded hover:bg-red-50">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                            </svg>
                        </button>
                    </div>`
                ).join('');
            } else {
                sourcesList.innerHTML = '<div class="text-center py-10 text-gray-400"><p>No sources indexed yet.</p><button onclick="showSection(\'add-source\')" class="text-primary-500 hover:underline mt-2 text-sm">Add one now</button></div>';
            }
        }
    } catch (e) {
        console.error('Failed to load stats:', e);
        Toast.show('Failed to load stats', 'error');
    }
}

async function handleScrape(e) {
    e.preventDefault();
    const btn = document.getElementById('scrapeBtn');
    const url = document.getElementById('urlInput').value;
    const name = document.getElementById('nameInput').value || url;

    // UI Loading State
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner mr-2"></div> Processing...';

    try {
        const res = await fetch(CONFIG.endpoints.scrape, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url,
                name,
                crawl_depth: parseInt(document.getElementById('crawlDepth').value),
                max_pages: parseInt(document.getElementById('maxPages').value)
            })
        });
        const data = await res.json();

        if (data.success) {
            Toast.show(`Successfully added ${data.chunks} chunks!`, 'success');
            document.getElementById('urlInput').value = '';
            document.getElementById('nameInput').value = '';
            loadStats();
        } else {
            Toast.show(data.error || 'Scraping failed', 'error');
        }
    } catch (e) {
        Toast.show(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function handleSearch(e) {
    e.preventDefault();
    const btn = document.getElementById('searchBtn');
    const resultsContainer = document.getElementById('searchResults');
    const query = document.getElementById('searchInput').value;

    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div>';

    // Skeleton Loading
    resultsContainer.innerHTML = Array(3).fill(0).map(() => `
        <div class="p-4 bg-white/40 rounded-lg border border-gray-200/50">
            <div class="h-4 bg-gray-200 rounded w-1/4 mb-3 skeleton"></div>
            <div class="h-3 bg-gray-200 rounded w-full mb-2 skeleton"></div>
            <div class="h-3 bg-gray-200 rounded w-5/6 skeleton"></div>
        </div>
    `).join('');

    try {
        const res = await fetch(CONFIG.endpoints.search, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: parseInt(document.getElementById('searchTopK').value) })
        });
        const data = await res.json();

        if (data.success && data.results.length > 0) {
            resultsContainer.innerHTML = data.results.map((r, i) => `
                <div class="p-5 glass rounded-xl hover:shadow-md transition fade-in" style="animation-delay: ${i * 0.1}s">
                    <div class="flex justify-between items-center mb-3">
                        <span class="text-xs font-bold px-2 py-1 rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300">
                            ${r.source}
                        </span>
                        <span class="text-xs text-gray-500 font-mono px-2 py-1 rounded bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300">Score: ${(r.score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed prose dark:prose-invert max-w-none">${marked.parse(r.text)}</div>
                </div>
            `).join('');
        } else {
            resultsContainer.innerHTML = '<div class="text-center py-8 text-gray-500">No relevant results found.</div>';
        }
    } catch (e) {
        resultsContainer.innerHTML = `<div class="p-4 bg-red-50 text-red-600 rounded-lg border border-red-100">${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Search';
    }
}

// Global scope for delete (called from HTML string)
window.deleteSource = async (sourceName) => {
    if (!confirm(`Are you sure you want to delete "${sourceName}"?`)) return;

    try {
        const res = await fetch(CONFIG.endpoints.deleteSource, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: sourceName })
        });
        const data = await res.json();

        if (data.success) {
            Toast.show(`Deleted ${sourceName}`, 'success');
            loadStats();
        } else {
            Toast.show(data.error, 'error');
        }
    } catch (e) {
        Toast.show(e.message, 'error');
    }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Init core systems
    ThemeManager.init();
    initNavigation();
    Toast.init();
    loadStats();

    // Event Listeners
    document.getElementById('themeToggleBtn')?.addEventListener('click', () => ThemeManager.toggle());
    document.getElementById('scrapeForm')?.addEventListener('submit', handleScrape);
    document.getElementById('searchForm')?.addEventListener('submit', handleSearch);

    // --- File Upload Handler ---
    const initFileUpload = () => {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const uploadStatus = document.getElementById('uploadStatus');
        const uploadFileName = document.getElementById('uploadFileName');

        if (!dropZone || !fileInput) return;

        // Click to browse
        dropZone.addEventListener('click', () => fileInput.click());

        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('border-indigo-500', 'bg-indigo-50/50');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('border-indigo-500', 'bg-indigo-50/50');
            });
        });

        // Handle drop
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        // Handle file input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        async function handleFileUpload(file) {
            // Show loading state
            uploadStatus.classList.remove('hidden');
            uploadFileName.textContent = `Uploading ${file.name}...`;
            dropZone.classList.add('pointer-events-none', 'opacity-70');

            const formData = new FormData();
            formData.append('file', file);
            formData.append('name', file.name);

            try {
                const res = await fetch(CONFIG.endpoints.upload, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.success) {
                    Toast.show(`Uploaded ${file.name}: ${data.chunks} chunks added!`, 'success');
                    loadStats();
                } else {
                    Toast.show(data.error || 'Upload failed', 'error');
                }
            } catch (e) {
                Toast.show(e.message, 'error');
            } finally {
                // Reset UI
                uploadStatus.classList.add('hidden');
                dropZone.classList.remove('pointer-events-none', 'opacity-70');
                fileInput.value = '';
            }
        }
    };
    initFileUpload();

    // --- Chat System with Conversational Memory ---
    const ChatSystem = {
        history: [], // Array of {role: 'user'|'assistant', content: '...'}
        container: null,
        placeholder: null,

        init() {
            this.container = document.getElementById('chatMessages');
            this.placeholder = document.getElementById('chatPlaceholder');

            const form = document.getElementById('dashAnswerForm');
            const newChatBtn = document.getElementById('newChatBtn');

            if (!form || !this.container) return;

            form.addEventListener('submit', (e) => this.handleSubmit(e));
            newChatBtn?.addEventListener('click', () => this.reset());
        },

        async handleSubmit(e) {
            e.preventDefault();
            const input = document.getElementById('dashQuestionInput');
            const btn = document.getElementById('dashSendBtn');
            const question = input.value.trim();

            if (!question) return;

            // Hide placeholder
            if (this.placeholder) this.placeholder.style.display = 'none';

            // Add user message to UI
            this.addMessage('user', question);
            input.value = '';

            // Add user message to history
            this.history.push({ role: 'user', content: question });

            // Show typing indicator
            const typingId = this.addTypingIndicator();

            // Disable button
            const originalBtnHTML = btn.innerHTML;
            btn.disabled = true;

            try {
                const settings = JSON.parse(localStorage.getItem('ragSettings') || '{}');
                settings.googleModel = 'gemini-2.5-flash';

                const res = await fetch(CONFIG.endpoints.answer, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question,
                        top_k: 3,
                        settings,
                        history: this.history.slice(0, -1) // Send history without current question
                    })
                });
                const data = await res.json();

                // Remove typing indicator
                this.removeTypingIndicator(typingId);

                if (data.success) {
                    // Add assistant message
                    this.addMessage('assistant', data.answer, data.contexts);
                    this.history.push({ role: 'assistant', content: data.answer });
                } else {
                    this.addMessage('error', data.error || 'Failed to get answer');
                }
            } catch (err) {
                this.removeTypingIndicator(typingId);
                this.addMessage('error', err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalBtnHTML;
            }
        },

        addMessage(role, content, contexts = []) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `fade-in ${role === 'user' ? 'flex justify-end' : ''}`;

            if (role === 'user') {
                msgDiv.innerHTML = `
                    <div class="max-w-[80%] bg-indigo-600 text-white px-5 py-3 rounded-2xl rounded-br-md shadow-sm">
                        <p class="text-sm">${this.escapeHtml(content)}</p>
                    </div>
                `;
            } else if (role === 'assistant') {
                const contextHtml = contexts.length ? `
                    <details class="mt-3 group">
                        <summary class="flex items-center gap-1 cursor-pointer text-xs text-gray-400 hover:text-indigo-500 transition list-none">
                            <svg class="w-3 h-3 transform group-open:rotate-90 transition" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
                            ${contexts.length} sources
                        </summary>
                        <div class="mt-2 space-y-1 pl-3 border-l border-gray-200 dark:border-gray-700 text-xs text-gray-500">
                            ${contexts.map(c => `<div class="truncate">${this.escapeHtml(c.substring(0, 100))}...</div>`).join('')}
                        </div>
                    </details>
                ` : '';

                msgDiv.innerHTML = `
                    <div class="max-w-[85%] glass px-5 py-4 rounded-2xl rounded-bl-md shadow-sm">
                        <div class="text-sm text-gray-800 dark:text-gray-100 leading-relaxed prose dark:prose-invert max-w-none">${marked.parse(content)}</div>
                        ${contextHtml}
                    </div>
                `;
            } else if (role === 'error') {
                msgDiv.innerHTML = `
                    <div class="max-w-[85%] bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-5 py-3 rounded-2xl">
                        <p class="text-sm">${this.escapeHtml(content)}</p>
                    </div>
                `;
            }

            this.container.appendChild(msgDiv);
            this.scrollToBottom();
        },

        addTypingIndicator() {
            const id = 'typing-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = 'fade-in';
            div.innerHTML = `
                <div class="glass px-5 py-4 rounded-2xl rounded-bl-md inline-flex items-center gap-2">
                    <div class="flex gap-1">
                        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
                        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
                        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
                    </div>
                    <span class="text-sm text-gray-400">Thinking...</span>
                </div>
            `;
            this.container.appendChild(div);
            this.scrollToBottom();
            return id;
        },

        removeTypingIndicator(id) {
            document.getElementById(id)?.remove();
        },

        scrollToBottom() {
            this.container.scrollTop = this.container.scrollHeight;
        },

        reset() {
            this.history = [];
            this.container.innerHTML = '';
            if (this.placeholder) {
                this.placeholder.style.display = 'block';
                this.container.appendChild(this.placeholder);
            }
            Toast.show('Started a new conversation', 'info');
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    };
    ChatSystem.init();

    // Settings Logic
    const initSettings = () => {
        const settings = JSON.parse(localStorage.getItem('ragSettings') || '{"provider":"ollama","ollamaModel":"qwen3:4b"}');
        const providerSelect = document.getElementById('modelProvider');
        if (!providerSelect) return;

        providerSelect.value = settings.provider;
        document.getElementById('ollamaModel').value = settings.ollamaModel || 'qwen3:4b';
        document.getElementById('googleApiKey').value = settings.googleApiKey || '';

        const toggleFields = () => {
            const isOllama = providerSelect.value === 'ollama';
            document.getElementById('ollamaSettings').style.display = isOllama ? 'block' : 'none';
            document.getElementById('googleSettings').style.display = isOllama ? 'none' : 'block';
        };

        providerSelect.addEventListener('change', toggleFields);
        toggleFields();

        document.getElementById('settingsForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const newSettings = {
                provider: providerSelect.value,
                ollamaModel: document.getElementById('ollamaModel').value,
                googleApiKey: document.getElementById('googleApiKey').value
            };
            localStorage.setItem('ragSettings', JSON.stringify(newSettings));
            Toast.show('Settings saved successfully', 'success');
        });
    };
    initSettings();
});
