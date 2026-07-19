const API_URL = "http://127.0.0.1:8000";

// Single Prediction
async function predictSingle() {
    const text = document.getElementById('reviewText').value.trim();
    
    if (!text) {
        alert("Please enter a review!");
        return;
    }

    const btn = document.querySelector('.predict-section button');
    btn.textContent = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        showResult(data);

    } catch (error) {
        alert("API not running! Start: uvicorn main:app --reload");
    } finally {
        btn.textContent = "Analyze Sentiment";
        btn.disabled = false;
    }
}

function showResult(data) {
    const resultDiv   = document.getElementById('result');
    const badgeDiv    = document.getElementById('sentiment-badge');
    const confText    = document.getElementById('conf-text');
    const confBar     = document.getElementById('conf-bar');

    resultDiv.classList.remove('hidden');

    const isPositive = data.sentiment === "Positive";

    badgeDiv.textContent = isPositive
        ? "✅ POSITIVE Review"
        : "❌ NEGATIVE Review";

    badgeDiv.className = isPositive
        ? "positive-badge"
        : "negative-badge";

    confText.textContent = `${data.confidence}%`;
    confBar.style.width  = `${data.confidence}%`;
    confBar.style.background = isPositive
        ? "linear-gradient(90deg, #2E7D32, #66BB6A)"
        : "linear-gradient(90deg, #C62828, #EF5350)";
}

// Batch Prediction
async function predictBatch() {
    const text  = document.getElementById('batchText').value.trim();
    const lines = text.split('\n').filter(l => l.trim() !== '');

    if (lines.length === 0) {
        alert("Please enter at least one review!");
        return;
    }

    const btn = document.querySelector('.batch-section button');
    btn.textContent = "Analyzing...";
    btn.disabled = true;

    const reviews = lines.map(l => ({ text: l.trim() }));

    try {
        const response = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reviews)
        });

        const data = await response.json();
        showBatchResults(data.results);

    } catch (error) {
        alert("API not running! Start: uvicorn main:app --reload");
    } finally {
        btn.textContent = "Analyze All Reviews";
        btn.disabled = false;
    }
}

function showBatchResults(results) {
    const container = document.getElementById('batch-results');
    container.innerHTML = '';

    // Summary
    const positive = results.filter(r => r.sentiment === 'Positive').length;
    const negative = results.length - positive;

    container.innerHTML = `
        <div style="margin-top:16px; padding:12px; 
                    background:#f0f4ff; border-radius:8px;
                    display:flex; gap:20px;">
            <span>✅ Positive: <strong>${positive}</strong></span>
            <span>❌ Negative: <strong>${negative}</strong></span>
            <span>📊 Total: <strong>${results.length}</strong></span>
        </div>
    `;

    // Individual results
    results.forEach(r => {
        const isPos = r.sentiment === 'Positive';
        const div   = document.createElement('div');
        div.className = `batch-item ${isPos ? 'positive' : 'negative'}`;
        div.innerHTML = `
            <span class="batch-text">
                ${r.text.substring(0, 80)}
                ${r.text.length > 80 ? '...' : ''}
            </span>
            <div>
                <span class="batch-badge">
                    ${isPos ? '✅ Positive' : '❌ Negative'}
                </span>
                <div style="font-size:0.8rem; color:#888; 
                            margin-top:4px; text-align:center;">
                    ${r.confidence}%
                </div>
            </div>
        `;
        container.appendChild(div);
    });
}