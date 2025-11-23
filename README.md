# Home Studio Recording Q&A System

AI-powered question answering system for the "Home Studio Recording: The Complete Guide" book. Built with Flask, Gemini API, and semantic search.

## 🚀 Live Demo
[Deploy on Render](#deployment-on-render)

## ✨ Features

- 💬 Natural language question answering about home studio recording
- 🔍 Semantic search across book content (chapters, paragraphs, key terms)
- 🎨 Modern, user-friendly UI with glass morphism effects
- ⚡ Fast responses with intelligent caching
- 📊 Relevance-scored source passages
- 🌐 Easy deployment to cloud platforms

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
# Windows PowerShell
$env:GEMINI_API_KEY='your_gemini_api_key'

# Linux / macOS
export GEMINI_API_KEY='your_gemini_api_key'
```

Get a free Gemini API key: https://aistudio.google.com/app/apikey

### 3. Run the Application
```bash
python app.py
```

Visit http://localhost:5000

## Deployment on Render

1. **Fork this repository** to your GitHub account

2. **Create a new Web Service** on [Render](https://render.com)
   - Connect your GitHub repository
   - Use the following settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`

3. **Add Environment Variable** in Render dashboard:
  - `GEMINI_API_KEY` - Your Gemini API key

4. **Deploy!** Render will automatically deploy your app

## Project Structure

```
pdf_book_main/
├── app.py                    # Flask web application
├── question_answering.py     # QA engine with semantic search
├── api_backends.py          # API backend handlers (Gemini, GitHub)
├── dataset_loader.py        # Dataset loading utilities
├── processed_data.json      # Pre-processed book content
├── qa_history.jsonl         # Cached Q&A responses
├── requirements.txt         # Python dependencies
├── runtime.txt             # Python version for deployment
├── Procfile               # Deployment configuration
├── templates/
│   └── index.html        # Web UI
├── static/
│   ├── css/
│   │   └── styles.css   # Styling with glass effects
│   └── js/
│       └── app.js       # Frontend JavaScript
└── datasets/
    ├── chapters.csv
    ├── paragraphs.csv
    ├── key_terms.csv
    └── tables_summary.csv
```

## API Endpoints

### POST /api/ask
Ask a question about the book.

**Request:**
```json
{
  "question": "What is mixing?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is mixing?",
  "answer": "Mixing is the process of combining...",
  "sources": [
    {
      "text": "Relevant passage text",
      "page_number": 276,
      "chapter_title": "Chapter 8",
      "score": 0.85
    }
  ],
  "backend_used": "gemini",
  "rate_limited": false
}
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY` - Gemini API key
- `PORT` - Server port (Render sets automatically)

### Search Depth Options
- **Quick** (3 passages) - Fast answers
- **Medium** (5 passages) - Balanced results
- **Deep** (7 passages) - More context
- **Comprehensive** (10 passages) - Maximum coverage

## Tech Stack

- **Backend:** Python, Flask
- **AI:** Google Gemini API (gemini-2.5-flash-lite)
- **Search:** Gemini Embeddings API (text-embedding-004)
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render + Gunicorn

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
python app.py

# The app runs on http://localhost:5000
```

## Git Commands for GitHub

```bash
# Initialize git repository (if not already done)
cd c:\Users\ASUS\Desktop\pdf_book_main
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Home Studio Recording Q&A System"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this project for your own Q&A systems!

---

**Built with ❤️ for music producers and home studio enthusiasts**
