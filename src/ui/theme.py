"""Shared UI constants and custom CSS for Gradio."""

CUSTOM_CSS = """
/* ---- global ---- */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.dark .gradio-container {
    background: #1e1e2e !important;
}

/* ---- header ---- */
#app-header {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
#app-header h1 {
    margin-bottom: 0.3rem;
    font-size: 1.8rem;
}

/* ---- generation result card ---- */
.result-card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
}

/* ---- song list sidebar ---- */
.song-list {
    max-height: 500px;
    overflow-y: auto;
}

/* ---- status badges ---- */
.badge-trained  { color: #2ecc71; font-weight: 600; }
.badge-training { color: #f39c12; font-weight: 600; }
.badge-error    { color: #e74c3c; font-weight: 600; }
.badge-untrained { color: #95a5a6; }

/* ---- training progress ---- */
.progress-text { font-family: monospace; white-space: pre-wrap; }
"""
