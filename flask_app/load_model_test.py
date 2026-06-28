import os

# Fix for PyTorch DLL loading on Windows
torch_lib = r'C:\Users\kashi\miniconda3\envs\atlas\lib\site-packages\torch\lib'
conda_env = r'C:\Users\kashi\miniconda3\envs\atlas'
os.environ['PATH'] = torch_lib + ';' + conda_env + ';' + os.environ['PATH']
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("done loading model")