# CineMatch India: AI-Powered Movie Recommender

CineMatch India is a sophisticated, hybrid movie recommendation system designed specifically for Indian cinema. It blends traditional lexical search, deep semantic embeddings, collaborative filtering patterns, and LLM-generated justifications into a premium, "cinema-style" user experience.

**Links:**  
- [GitHub Repo](https://github.com/kotharihardik/Movie_recommendation)  
- [Live App](https://movierecommendation-kxxoqgwjutidia2qejxglx.streamlit.app/)  
- [Video Demo](https://www.youtube.com/watch?v=JTca_iBlRQQ)

## Key Features

- **Hybrid Ranking Engine**: Combines TF-IDF, Semantic Embeddings (Sentence-Transformers), and SVD-based collaborative patterns.
- **Star Power Heuristic**: A custom "Fame Score" that prioritizes mainstream cinema by analyzing billing-weighted cast and director frequency.
- **Vibe-Based Search**: Natural language "mood" queries (e.g., *"Gritty underworld thriller with high stakes"*) using vector similarity.
- **LLM Justifications**: Integrated with **Google Gemini** to provide personalized reasoning for every recommendation.
- **Premium UI**: A custom-styled, pure-black Streamlit interface designed for a high-end cinematic feel.
- **Benchmark Suite**: Comprehensive evaluation using Precision@K, Recall@K, MRR, MAP, and NDCG@K with stratified query sampling.

##  Technology Stack
- **Frontend**: Streamlit (with Custom CSS Injection)
- **Vector DB**: ChromaDB
- **Models**:
  - `all-MiniLM-L6-v2` (Embeddings)
  - `cross-encoder/stsb-roberta-base` (Reranking)
  - `gemini-1.5-flash` (Justifications)
- **Logic**: Scikit-Learn (TF-IDF, SVD, KNN), Pandas, NumPy

##  Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key

### 2. Installation
```bash
git clone <your-repo-url>
cd Movie_recommendation
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file or set the following environment variables:
```env
GEMINI_API_KEY=your_api_key_here
MOVIES_CSV=data/movies.csv
CHROMA_DB_PATH=./chroma_db
```

### 4. Running the App
```bash
streamlit run app.py
```

##  Evaluation
To run the quantitative evaluation suite and see the system's performance on the benchmark set:
```bash
python evaluation.py
```
This will report **NDCG@10**, **MAP@10**, and **ILD** (Intra-List Diversity) based on manual and stratified ground-truth data.

## Project Structure

- `app.py`: Main application orchestration and Streamlit UI.
- `recommend_engine.py`: The core hybrid recommendation logic.
- `llm_client.py`: Gemini API integration and justification logic.
- `data_pipeline.py`: Data cleaning, rich-text building, and ChromaDB indexing.
- `evaluation.py`: Performance metrics and benchmarking tools.
- `ui_components.py`: Custom CSS and theme enforcement.

---

##  License
This project was developed for educational purposes as part of the **SMAI** course.
