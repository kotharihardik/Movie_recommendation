#!/usr/bin/env python3
"""Quick accessibility check for embedding models used in tests.
Tries to load each model and encode a short sample to verify download/load/encode works.
Saves a summary to `model_access_check.txt`.
"""
import time
from sentence_transformers import SentenceTransformer
import traceback

MODELS = [
    'intfloat/e5-large-v2',
    'BAAI/bge-large-en-v1.5',
    'sentence-transformers/all-mpnet-base-v2',
    'intfloat/e5-base-v2',
    'all-MiniLM-L6-v2'
]

SAMPLE_TEXT = "A test movie overview about friendship and revenge."

out_lines = []
for m in MODELS:
    print(f"Checking: {m}")
    start = time.time()
    status = 'unknown'
    err = None
    try:
        # try MPS then CPU
        try:
            model = SentenceTransformer(m, device='mps')
        except Exception:
            model = SentenceTransformer(m, device='cpu')
        # prepare sample (apply e5 prefix if model name contains 'e5')
        text = SAMPLE_TEXT
        if 'e5' in m:
            text = 'query: ' + text
        emb = model.encode([text], show_progress_bar=False)
        duration = time.time() - start
        status = 'ok'
        out_lines.append(f"{m}\tOK\t{duration:.2f}s\n")
        print(f"  OK ({duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start
        status = 'error'
        tb = traceback.format_exc()
        out_lines.append(f"{m}\tERROR\t{duration:.2f}s\t{str(e).splitlines()[0]}\n")
        print(f"  ERROR after {duration:.2f}s: {e}")

# write results
with open('model_access_check.txt', 'w') as f:
    f.write('Model\tStatus\tTime\tNotes\n')
    for line in out_lines:
        f.write(line)

print('\nSummary written to model_access_check.txt')
