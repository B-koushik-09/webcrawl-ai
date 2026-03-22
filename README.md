# 🎓 VNRVJIET AI - Intelligent College Website Knowledge Assistant

An AI-powered web intelligence system that automatically scrapes, understands, and answers queries from college websites including all linked PDF documents.

![VNRVJIET AI](https://img.shields.io/badge/Version-1.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![React](https://img.shields.io/badge/React-19+-cyan)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-red)

## 🧠 Product Vision

To build an AI-powered web intelligence system that:
- Automatically scrapes and indexes entire public college websites
- Discovers, downloads, and extracts all linked PDF documents
- Provides instant, accurate, source-grounded answers to natural language questions
- Ensures responses are strictly based on official website data

## 🏗️ Architecture

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + Vite |
| Backend | Python FastAPI |
| Scraper | Firecrawl (Primary) / BeautifulSoup |
| PDF Processor | PyMuPDF |
| Embeddings | BGE-Small / Sentence Transformers |
| Vector DB | FAISS |
| AI Engine | RAG (Retrieval-Augmented Generation) |
| LLMs | Google Gemini 2.0 Flash (Primary), Mistral 7B, Ollama |

## 📁 Project Structure

```
ai-chat/
├── Backend/
│   ├── app.py                 # Main FastAPI application
│   ├── config.py              # Configuration settings
│   ├── modules/               # Core logic (Scraper, RAG, Embeddings)
│   ├── data/                  # Raw scraped data and PDFs (Git ignored)
│   ├── cleaned_pages/         # Processed markdown content (Git ignored)
│   ├── vector_store/          # FAISS index files (Git ignored)
│   └── requirements.txt       # Python dependencies
├── frontend/                  # React Frontend
├── .gitignore                 # Unified git ignore rules
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Environment Configuration

Create a `.env` file in the `Backend/` directory:

```env
# Web Crawling
FIRECRAWL_API_KEY=your_firecrawl_key

# LLM Tier 1: Google Gemini (primary)
GEMINI_API_KEY=your_gemini_key

# LLM Tier 2: Hugging Face (secondary)
HF_TOKEN=your_huggingface_token
```

### 2. Backend Setup

```bash
cd Backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing Your Setup

A utility script is provided to verify your Gemini API key:

```bash
cd Backend
python gemini_test.py
```

## 📖 Features

### 1. Intelligent Indexing
The system uses **Firecrawl** to deeply map college websites, capturing both standard HTML pages and complex PDF documents.

### 2. Topic-Based RAG
Unlike simple keyword search, our RAG engine understands the context of college data (e.g., distinguishing between "CSE Syllabus" and "ECE Syllabus").

### 3. Source Grounding
Every answer includes citations with links to the original sources, ensuring 100% accuracy and trust.

## 🛠️ Deployment

### Railway (Recommended for Backend)
1. Link your GitHub repo to Railway.
2. Add your environment variables in the Railway dashboard.
3. Railway will automatically detect the `Backend/` folder and deploy using the `requirements.txt`.

### Vercel / Netlify (For Frontend)
1. Deploy the `frontend/` folder.
2. Set `VITE_API_URL` to your Backend URL.

## 📝 License

This project is licensed under the MIT License.

---

Built with ❤️ for VNRVJIET Students
