# aayu-find

creating of the conda env

conda create -n ayur_fyp python=3.12
conda activate aayu_find

pip install torch --index-url https://download.pytorch.org/whl/cu128

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload