// DOM Elements
const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const answerText = document.getElementById('answerText');
const sourcesContainer = document.getElementById('sourcesContainer');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const backButton = document.getElementById('backButton');

// Event Listeners
askButton.addEventListener('click', handleAskQuestion);
backButton.addEventListener('click', handleBackToSearch);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleAskQuestion();
    }
});

// Main Functions
async function handleAskQuestion() {
    const question = questionInput.value.trim();
    const selectedRadio = document.querySelector('input[name="search-depth"]:checked');
    const topK = parseInt(selectedRadio.value);

    if (!question) {
        showError('Please enter a question');
        return;
    }

    // Show loader, hide other sections
    loader.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    askButton.disabled = true;

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: topK
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get answer');
        }

        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        loader.style.display = 'none';
        askButton.disabled = false;
    }
}

function displayResults(data) {
    // Display answer with special styling if rate limited
    answerText.textContent = data.answer;
    
    // Add visual indicator for rate limit
    if (data.rate_limited) {
        answerText.style.backgroundColor = '#fff3cd';
        answerText.style.borderLeft = '4px solid #ffc107';
        answerText.style.color = '#856404';
    } else {
        answerText.style.backgroundColor = '#f8f9fa';
        answerText.style.borderLeft = '4px solid #667eea';
        answerText.style.color = '';
    }

    // Display sources
    sourcesContainer.innerHTML = '';
    data.sources.forEach((source, index) => {
        const sourceCard = createSourceCard(source, index + 1);
        sourcesContainer.appendChild(sourceCard);
    });

    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createSourceCard(source, index) {
    const card = document.createElement('div');
    card.className = 'source-card';

    // Format location
    let location = '';
    if (source.chapter_number) {
        location = `CHAPTER${source.chapter_number}, page ${source.page_number}`;
    } else if (source.chapter_title) {
        location = `${source.chapter_title}, page ${source.page_number}`;
    } else {
        location = `Page ${source.page_number}`;
    }

    // Convert score to relevance level
    let relevanceLevel = '';
    let relevanceClass = '';
    const score = parseFloat(source.score);
    
    if (score >= 0.7) {
        relevanceLevel = 'Highly Relevant';
        relevanceClass = 'relevance-high';
    } else if (score >= 0.5) {
        relevanceLevel = 'Very Relevant';
        relevanceClass = 'relevance-very';
    } else if (score >= 0.35) {
        relevanceLevel = 'Relevant';
        relevanceClass = 'relevance-medium';
    } else if (score >= 0.2) {
        relevanceLevel = 'Somewhat Relevant';
        relevanceClass = 'relevance-low';
    } else {
        relevanceLevel = 'Related';
        relevanceClass = 'relevance-minimal';
    }

    card.innerHTML = `
        <div class="source-header">
            <span class="source-location">${location}</span>
            <span class="source-score ${relevanceClass}">${relevanceLevel}</span>
        </div>
        <div class="source-text">${escapeHtml(source.text)}</div>
    `;

    return card;
}

function handleBackToSearch() {
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    questionInput.focus();
}

function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    loader.style.display = 'none';
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        errorSection.style.display = 'none';
    }, 5000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initial focus
questionInput.focus();
