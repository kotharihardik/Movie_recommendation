#!/usr/bin/env python3
"""
Test script for 15 popular Bollywood movies
Tests sentence transformer models on diverse Bollywood films
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm
import os
from datetime import datetime

# Configuration
MODELS = [
    # Tier 1 (best-first try)
    'intfloat/e5-large-v2',
    'BAAI/bge-large-en-v1.5',
    # Tier 2 (strong upgrade)
    'sentence-transformers/all-mpnet-base-v2',
    'intfloat/e5-base-v2',
    # Baseline
    'all-MiniLM-L6-v2'
]

# 15 Movies to test (exact names from dataset)
TEST_MOVIES = [
    "3 Idiots",
    "Hera Pheri",
    "Bhool Bhulaiyaa 2",
    "RRR",
    "Drishyam 2",
    "Shreemaan Aashique",
    "Happy New Year",
    "Kabir Singh",
    "Bhoothnath",
    "Munna Bhai M.B.B.S.",
    "Singham Again",
    "Panneer Pushpangal",
    "Krishna and Kamsa",
    "Ra.One",
    "K.G.F: Chapter 1"
]

TOP_N = 10
MIN_VOTES = 1  # Lowered to include more movies like Shreemaan Aashique, Krishna and Kamsa, Panneer Pushpangal


def load_data():
    """Load the movie dataset"""
    print("Loading dataset...")
    df = pd.read_csv('tmdb_large_dataset.csv')
    # Filter for movies with sufficient votes
    df = df[df['vote_count'] >= MIN_VOTES].copy()
    print(f"  Loaded {len(df)} movies with vote_count >= {MIN_VOTES}")
    return df


def get_test_movies(df):
    """Get the 15 test movies from dataset"""
    test_movies = []
    for movie_name in TEST_MOVIES:
        matches = df[df['title'] == movie_name]
        if len(matches) > 0:
            test_movies.append(matches.iloc[0].to_dict())
            print(f"  ✓ {movie_name}")
        else:
            print(f"  ✗ {movie_name} NOT FOUND")
    
    print(f"\nFound {len(test_movies)}/{len(TEST_MOVIES)} test movies")
    return test_movies


def make_description(movie, model_name=None):
    """Create description from movie overview.

    For some models (e.g. E5) a symmetric "query: " prefix improves
    semantic similarity for paragraph-level retrieval. If `model_name`
    contains 'e5' we prepend the prefix.
    """
    overview = str(movie.get('overview', ''))
    title = str(movie.get('title', ''))
    text = f"{title}. {overview}"
    if model_name and ('e5-' in model_name or model_name.startswith('intfloat/e5')):
        return f"query: {text}"
    return text


def load_model(model_name):
    """Load sentence transformer model"""
    print(f"\n  Loading {model_name.split('/')[-1]}...")
    try:
        model = SentenceTransformer(model_name, device='mps')
        print(f"  Model loaded successfully")
        return model
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None


def encode_descriptions(model, descriptions):
    """Encode movie descriptions to embeddings"""
    print(f"  Encoding {len(descriptions)} movie descriptions...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def get_top_n_recommendations(query_embedding, all_embeddings, movies_df, n=TOP_N, exclude_idx=None):
    """Get top N recommendations using cosine similarity"""
    similarities = np.dot(all_embeddings, query_embedding)
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    
    recommendations = []
    for idx in sorted_indices:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        if len(recommendations) >= n:
            break
        
        movie_idx = idx
        sim_score = float(similarities[idx])
        movie_data = movies_df.iloc[movie_idx].to_dict()
        movie_data['similarity_score'] = sim_score
        recommendations.append(movie_data)
    
    return recommendations


def format_movie_info(movie):
    """Format movie information"""
    title = movie.get('title', 'Unknown')
    year = movie.get('release_date', '')[:4] if pd.notna(movie.get('release_date')) else 'N/A'
    rating = movie.get('vote_average', 0)
    overview = movie.get('overview', '')[:200]
    
    info = f"{title} ({year}) - Rating: {rating}/10\n"
    info += f"Overview: {overview}...\n"
    return info


def generate_report(model_name, test_movies, all_movies_df, all_embeddings):
    """Generate test report"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.split('/')[-1]}")
    print(f"{'='*60}")
    
    model = load_model(model_name)
    if model is None:
        print(f"Skipping {model_name}")
        return
    
    # Reset index to ensure alignment
    all_movies_df = all_movies_df.reset_index(drop=True)
    
    # Prepare descriptions for encoding (allow model-specific prefixes)
    all_descriptions = [make_description(movie, model_name=model_name) for movie in all_movies_df.to_dict('records')]
    print(f"\n  Encoding {len(all_descriptions)} movies...")
    embeddings = encode_descriptions(model, all_descriptions)
    
    # Create output
    output_file = f"model_test_results/test_15movies_{model_name.split('/')[-1]}.txt"
    os.makedirs('model_test_results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"Movie Recommendation System - Model Test\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Movies: {len(test_movies)}\n")
        f.write(f"Dataset Size: {len(all_movies_df)}\n")
        f.write("="*60 + "\n\n")
        
        # Test each movie
        for test_idx, test_movie in enumerate(tqdm(test_movies, desc="Testing movies"), 1):
            # Find position of test movie in full dataset (using reset index)
            matching_rows = all_movies_df[all_movies_df['title'] == test_movie['title']]
            if len(matching_rows) == 0:
                print(f"  Warning: {test_movie['title']} not found in filtered dataset")
                continue
            
            test_movie_idx = matching_rows.index[0]
            
            # Get recommendations
            recommendations = get_top_n_recommendations(
                embeddings[test_movie_idx],
                embeddings,
                all_movies_df,
                n=TOP_N,
                exclude_idx=test_movie_idx
            )
            
            # Write test movie
            f.write(f"\n{'='*60}\n")
            f.write(f"TEST MOVIE {test_idx}: {test_movie['title']}\n")
            f.write(f"{'='*60}\n")
            f.write(format_movie_info(test_movie))
            f.write(f"\nGenres: {test_movie.get('genres', 'N/A')}\n")
            f.write(f"Keywords: {test_movie.get('keywords', 'N/A')}\n")
            
            # Write recommendations
            f.write(f"\nTOP {TOP_N} RECOMMENDATIONS:\n")
            f.write("-" * 60 + "\n")
            
            for rank, rec in enumerate(recommendations, 1):
                score = rec.get('similarity_score', 0)
                f.write(f"\n{rank}. {rec['title']} - Score: {score:.4f}\n")
                f.write(f"   Year: {str(rec.get('release_date', ''))[:4]}\n")
                f.write(f"   Rating: {rec.get('vote_average', 'N/A')}/10\n")
                f.write(f"   Overview: {str(rec.get('overview', ''))[:150]}...\n")
                f.write(f"   Keywords: {rec.get('keywords', 'N/A')}\n")
            
            f.write("\n" + " "*60 + "\n")  # Space between movies
    
    print(f"\n✓ Report saved: {output_file}")
    return output_file


def main():
    """Main test function"""
    print("="*60)
    print("BOLLYWOOD MOVIE RECOMMENDATION - MODEL TEST")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Get test movies
    print("\nLoading test movies...")
    test_movies = get_test_movies(df)
    
    if len(test_movies) < 15:
        print(f"\nWARNING: Only found {len(test_movies)}/15 test movies")
    
    # Test each model
    for model_name in MODELS:
        try:
            generate_report(model_name, test_movies, df, None)
        except Exception as e:
            print(f"ERROR testing {model_name}: {e}")
    
    print("\n" + "="*60)
    print("✓ Testing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
