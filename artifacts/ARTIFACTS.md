# Preprocessed artifacts — commit these to the repo after one successful run.
#
# After full preprocess + train_ltr:
#   python scripts/lock_artifacts.py
#   git add artifacts/
#   git commit -m "Lock offline preprocess artifacts"
#
# Large files (bge_embeddings.npy, faiss.index) need Git LFS — see .gitattributes.
#
# Reuse without spending Gemini credits:
#   set REDROB_SKIP_GEMINI=1
#   python scripts/preprocess.py --skip-llm
#
# Only re-call Gemini intentionally:
#   python scripts/preprocess.py --step labels --force-llm
