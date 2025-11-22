"""Flask web application for Home Studio Recording QA system."""

from flask import Flask, request, jsonify, render_template
from question_answering import BookQAService
import os

app = Flask(__name__)

# Initialize QA service (lazy loading)
qa_service = None


def get_qa_service():
    """Lazy-load QA service to avoid startup delays."""
    global qa_service
    if qa_service is None:
        qa_service = BookQAService(
            backend_type="gemini",
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    return qa_service


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle question requests."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        top_k = int(data.get('top_k', 3))
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if not (1 <= top_k <= 10):
            return jsonify({'error': 'top_k must be between 1 and 10'}), 400
        
        # Get QA service and answer
        service = get_qa_service()
        result = service.answer_question(question, top_k=top_k)
        
        # Check if answer contains rate limit warning
        is_rate_limited = result['answer'].startswith('⚠️ API rate limit')
        
        # Format response
        response = {
            'question': result['question'],
            'answer': result['answer'],
            'rate_limited': is_rate_limited,
            'sources': [
                {
                    'page_number': src['page_number'],
                    'chapter_number': src.get('chapter_number'),
                    'chapter_title': src.get('chapter_title'),
                    'score': round(src['score'], 3),
                    'text': src['text'][:200] + '...' if len(src['text']) > 200 else src['text']
                }
                for src in result['passages']
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
