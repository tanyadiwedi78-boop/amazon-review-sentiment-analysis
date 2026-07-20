const API_URL = "http://127.0.0.1:8000"

// Example chips
function fillExample(text) {
    document.getElementById('reviewText').value = text
}

// Single Prediction
async function predictSingle() {
    const text = document.getElementById('reviewText').value.trim()

    if (!text) {
        alert("Please enter a review!")
        return
    }

    // Show loading
    document.getElementById('placeholder').classList.add('hidden')
    document.getElementById('result-content').classList.add('hidden')
    document.getElementById('loading').classList.remove('hidden')

    const btn = document.getElementById('analyzeBtn')
    btn.textContent = "Analyzing..."
    btn.disabled = true

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })

        const data = await response.json()
        showResult(data)

    } catch (error) {
        alert("API not running! Start: uvicorn backend.main:app --reload")
        document.getElementById('loading').classList.add('hidden')
        document.getElementById('placeholder').classList.remove('hidden')
    } finally {
        btn.textContent = "Analyze Sentiment"
        btn.disabled = false
    }
}

function showResult(data) {
    // Hide loading, show result
    document.getElementById('loading').classList.add('hidden')
    document.getElementById('result-content').classList.remove('hidden')

    const isPositive = data.sentiment === "Positive"

    // Sentiment badge
    const badge = document.getElementById('sentiment-badge')
    badge.textContent = isPositive
        ? "✅ POSITIVE Review"
        : "❌ NEGATIVE Review"
    badge.className = `sentiment-badge ${isPositive ? 'positive-badge' : 'negative-badge'}`

    // Confidence
    document.getElementById('conf-text').textContent = `${data.confidence}%`
    const bar = document.getElementById('conf-bar')
    bar.style.width = `${data.confidence}%`
    bar.style.background = isPositive
        ? "linear-gradient(90deg, #2E7D32, #66BB6A)"
        : "linear-gradient(90deg, #C62828, #EF5350)"

    // AI Insights paragraph
    if (data.ai_insights) {
        document.getElementById('ai-paragraph').textContent = data.ai_insights
    }
}

// Batch Prediction
async function predictBatch() {
    const text  = document.getElementById('batchText').value.trim()
    const lines = text.split('\n').filter(l => l.trim() !== '')

    if (lines.length === 0) {
        alert("Please enter at least one review!")
        return
    }

    const btn = document.querySelector('.batch-container button')
    btn.textContent = "Analyzing..."
    btn.disabled = true

    const reviews = lines.map(l => ({ text: l.trim() }))

    try {
        const response = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reviews)
        })

        const data = await response.json()
        showBatchResults(data.results)

    } catch (error) {
        alert("API not running!")
    } finally {
        btn.textContent = "Analyze All Reviews"
        btn.disabled = false
    }
}

function showBatchResults(results) {
    const container = document.getElementById('batch-results')
    container.innerHTML = ''

    // Summary bar
    const positive = results.filter(r => r.sentiment === 'Positive').length
    const negative = results.length - positive

    container.innerHTML = `
        <div style="display:flex; gap:20px; margin:16px 0;
                    padding:14px 18px; background:#f8faff;
                    border-radius:10px; border:1px solid #e3eaf5;">
            <span>✅ Positive: <strong>${positive}</strong></span>
            <span>❌ Negative: <strong>${negative}</strong></span>
            <span>📊 Total: <strong>${results.length}</strong></span>
        </div>
    `

    results.forEach(r => {
        const isPos = r.sentiment === 'Positive'
        const div   = document.createElement('div')
        div.className = `batch-item ${isPos ? 'positive' : 'negative'}`
        div.innerHTML = `
            <div class="batch-text">
                ${r.text.substring(0, 100)}${r.text.length > 100 ? '...' : ''}
            </div>
            <div style="text-align:right; min-width:100px;">
                <span class="batch-badge">
                    ${isPos ? '✅ Positive' : '❌ Negative'}
                </span>
                <div style="font-size:0.78rem; color:#999;
                            margin-top:4px;">${r.confidence}%</div>
            </div>
        `
        container.appendChild(div)
    })
}

// Enter key support
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        predictSingle()
    }
})