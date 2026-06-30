import nltk
import yaml
import os
import re
import pandas as pd
import contractions

from nltk.stem import WordNetLemmatizer
from src.logger import logging

# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# Chat Abbreviations
# ---------------------------------------------------

chat_words = {

    # General
    "asap": "as soon as possible",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "btw": "by the way",
    "idk": "i do not know",
    "imo": "in my opinion",
    "thx": "thanks",
    "msg": "message",

    # Banking
    "acct": "account",
    "acc": "account",
    "bk": "bank",
    "bal": "balance",
    "chk": "checking",
    "sav": "savings",
    "dep": "deposit",
    "wd": "withdrawal",
    "atm": "automated teller machine",

    # Credit cards
    "cc": "credit card",
    "ccs": "credit cards",
    "cvv": "card verification value",
    "apr": "annual percentage rate",
    "stmt": "statement",

    # Credit reporting
    "cr": "credit report",
    "cra": "credit reporting agency",
    "cb": "credit bureau",
    "eq": "equifax",
    "exp": "experian",
    "tu": "transunion",

    # Loans
    "amt": "amount",
    "loan amt": "loan amount",
    "emi": "equated monthly installment",
    "int": "interest",
    "refi": "refinance",
    "mod": "modification",

    # Mortgage
    "mtg": "mortgage",
    "heloc": "home equity line of credit",

    # Transactions
    "txn": "transaction",
    "txns": "transactions",
    "auth": "authorization",
    "authd": "authorized",

    # Customer service
    "cust": "customer",
    "svc": "service",
    "rep": "representative",
    "csr": "customer service representative",

    # Complaint
    "info": "information",
    "doc": "document",
    "docs": "documents",
    "resp": "response",
    "wrt": "with respect to",

    # Masked tokens
    "xxxx": "xxxx",
    "xx": "xx"
}



# ---------------------------------------------------
# Load Params
# ---------------------------------------------------

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

def load_data(file_path):

    df = pd.read_csv(file_path)

    logging.info("Processed raw data loaded")

    return df


# ---------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = contractions.fix(text)

    words = text.split()

    words = [
        chat_words[word]
        if word in chat_words
        else word
        for word in words
    ]

    text = " ".join(words)

    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    text = re.sub(r"http\S+|www\S+", " URL ", text)

    text = re.sub(r"\b\d{10,}\b", " PHONE ", text)

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------
# Apply preprocessing
# ---------------------------------------------------

def preprocess_dataframe(
    df,
    text_column
):

    df[text_column] = df[text_column].apply(preprocess_text)

    logging.info("Text preprocessing completed")

    return df


# ---------------------------------------------------
# Save
# ---------------------------------------------------

def save_processed_data(
    df,
    output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    logging.info("Processed data saved")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    try:
        params = load_params("params.yaml")

        input_path = params["data_preprocessing"]["input_path"]
        output_path = params["data_preprocessing"]["output_path"]
        text_column = params["data"]["text_column"]

        df = load_data(input_path)

        df = preprocess_dataframe(df, text_column)

        save_processed_data(df, output_path)

        logging.info("Data preprocessing completed successfully.")

    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()