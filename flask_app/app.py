import os 

import warnings
warnings.filterwarnings('error')

# NOW import everything else
import mlflow
import pickle
import pandas as pd
import time
import warnings
import re
import string
import dagshub
from flask import Flask, render_template, request
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from preprocessing_utiliy import preprocess_text
from load_model_test import embedding_model
import torch 


warnings.filterwarnings("ignore")

app = Flask(__name__)



# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "sahilkatiyar2024"
repo_name = "CFPB_Complaint_Intelligence_pipeline"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------

def load_latest_model(model_name):
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    
    production_versions = [v for v in versions if v.current_stage == "Production"]
    
    if production_versions:
        version = production_versions[0].version
    else:
        version = sorted(versions, key=lambda v: int(v.version))[-1].version

    uri = f"models:/{model_name}/{version}"
    print(f"Loading {model_name} version {version}")
    
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, "map_location": torch.device("cpu")})

    return mlflow.pyfunc.load_model(uri)



product_model = load_latest_model("Product_model")
print("done loading product model")

urgency_model = load_latest_model("urgent_model")



PRODUCT_MAPPING = {

    0: "Bank Accounts and Services",

    1: "Credit Card Services",

    2: "Credit Reporting",

    3: "Debt Collection",

    4: "Loans"

}

URGENCY_MAPPING = {

    0: "No",

    1: "Yes"

}


@app.route("/")
def home():

    return render_template(
        "index.html",
        product=None,
        urgency=None
    )
    

@app.route("/predict", methods=["POST"])

def predict():

    user_text = request.form["text"]

    # -----------------------------
    # Step 1 : Preprocess
    # -----------------------------

    cleaned_text = preprocess_text(user_text)

    # -----------------------------
    # Step 2 : Create Embedding
    # -----------------------------

    embedding = embedding_model.encode([cleaned_text])

    embedding_df = pd.DataFrame(embedding)

    import numpy as np

    # Product model (pytorch) - need argmax of logits
    product_pred_raw = product_model.predict(embedding_df)
    product_arr = np.array(product_pred_raw).flatten()
    print("product raw output:", product_arr)  # see what shape/values come out

    # If it returns logits (multiple values per class)
    if len(product_arr) > 1:
        product_val = int(np.argmax(product_arr))
    else:
        product_val = int(product_arr[0])

    # Urgency model (sklearn) - direct class label
    urgency_pred_raw = urgency_model.predict(embedding_df)
    urgency_val = int(np.array(urgency_pred_raw).flatten()[0])

    product = PRODUCT_MAPPING[product_val]
    urgency = URGENCY_MAPPING[urgency_val]

    return render_template(

        "index.html",

        product=product,

        urgency=urgency

    )
    
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )