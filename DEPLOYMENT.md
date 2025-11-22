# Home Studio Recording QA Website

A web application that answers questions about "Home Studio Recording: The Complete Guide" using AI-powered question answering.

## Features

- **Intelligent Question Answering**: Uses sentence transformers for semantic search and GitHub Models API for natural language answers
- **Source Citations**: Shows relevant passages from the book with page and chapter information
- **Modern UI**: Animated loader and responsive design with gradient backgrounds
- **Adjustable Retrieval**: Choose how many source passages to retrieve (1-10)

## Local Development

### Prerequisites

- Python 3.11+
- GitHub Models API token (free, unlimited usage)

### Setup

1. **Clone the repository**
   ```bash
   cd 3rd_attempt_book
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variable**
   ```bash
   # Windows PowerShell
   $env:GITHUB_TOKEN = "your_github_token_here"
   
   # Linux/Mac
   export GITHUB_TOKEN="your_github_token_here"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## Deployment

### Option 1: Render (Recommended - Free Tier)

1. Create account at [render.com](https://render.com)
2. Create new **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment Variable**: `GITHUB_TOKEN` = your token
5. Deploy (takes ~5 minutes)

### Option 2: Railway

1. Create account at [railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Add environment variable `GITHUB_TOKEN`
4. Deploy automatically

### Option 3: Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Set token: `heroku config:set GITHUB_TOKEN=your_token`
5. Deploy: `git push heroku main`

## File Structure

```
.
├── app.py                  # Flask application
├── question_answering.py   # QA logic and retrieval
├── api_backends.py         # GitHub Models API integration
├── dataset_loader.py       # Load processed book data
├── processed_data.json     # Indexed book content
├── templates/
│   └── index.html         # Main HTML template
├── static/
│   ├── css/
│   │   └── styles.css     # Styling with animations
│   └── js/
│       └── app.js         # Frontend logic
├── requirements.txt       # Python dependencies
├── Procfile              # For Render/Heroku deployment
└── runtime.txt           # Python version specification
```

## API Endpoints

- `GET /` - Main web interface
- `POST /api/ask` - Submit questions
  - Body: `{"question": "...", "top_k": 3}`
  - Response: `{"question": "...", "answer": "...", "sources": [...]}`
- `GET /health` - Health check endpoint

## Technologies

- **Backend**: Flask, Python 3.11
- **AI/ML**: Sentence Transformers (all-MiniLM-L6-v2), GitHub Models API (GPT-4o)
- **Frontend**: Vanilla JavaScript, CSS3 animations
- **Data**: Pre-processed book content with embeddings

## Environment Variables

- `GITHUB_TOKEN` - Required. Your GitHub Models API token
- `PORT` - Optional. Server port (default: 5000)

## Notes

- First question may take ~10 seconds due to model loading
- Subsequent questions are faster (~2-3 seconds)
- GitHub Models API is free with unlimited usage
- No GPU required - embeddings are pre-computed

## License

Educational use only. Book content is copyrighted.
