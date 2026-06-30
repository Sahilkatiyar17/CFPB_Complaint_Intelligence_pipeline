import os

# rest of your imports...
import pandas as pd
import re
import contractions

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# stop_words = set(stopwords.words('english'))

# # Keep negations because they are important
# stop_words = stop_words - {
#     'not', 'no', 'nor',
#     "don't", "didn't", "won't",
#     "isn't", "aren't"
# }

lemmatizer = WordNetLemmatizer()

chat_words = {

    # General chat abbreviations
    'asap': 'as soon as possible',
    'pls': 'please',
    'plz': 'please',
    'u': 'you',
    'ur': 'your',
    'btw': 'by the way',
    'idk': 'i do not know',
    'imo': 'in my opinion',
    'thx': 'thanks',
    'msg': 'message',

    # Banking
    'acct': 'account',
    'acc': 'account',
    'bk': 'bank',
    'bal': 'balance',
    'chk': 'checking',
    'sav': 'savings',
    'dep': 'deposit',
    'wd': 'withdrawal',
    'atm': 'automated teller machine',

    # Credit Cards
    'cc': 'credit card',
    'ccs': 'credit cards',
    'cvv': 'card verification value',
    'apr': 'annual percentage rate',
    'stmt': 'statement',

    # Credit Reporting
    'cr': 'credit report',
    'cra': 'credit reporting agency',
    'cb': 'credit bureau',
    'eq': 'equifax',
    'exp': 'experian',
    'tu': 'transunion',

    # Loans
    'amt': 'amount',
    'loan amt': 'loan amount',
    'emi': 'equated monthly installment',
    'int': 'interest',
    'refi': 'refinance',
    'mod': 'modification',

    # Mortgage
    'mtg': 'mortgage',
    'heloc': 'home equity line of credit',
    'escrow': 'escrow',

    # Transactions
    'txn': 'transaction',
    'txns': 'transactions',
    'auth': 'authorization',
    'authd': 'authorized',

    # Customer service
    'cust': 'customer',
    'svc': 'service',
    'rep': 'representative',
    'csr': 'customer service representative',

    # Debt collection
    'dc': 'debt collection',
    'ca': 'collection agency',

    # Identity / Personal Info
    'ssn': 'social security number',
    'dob': 'date of birth',
    'pii': 'personally identifiable information',

    # Complaint-related words
    'info': 'information',
    'doc': 'document',
    'docs': 'documents',
    'resp': 'response',
    'wrt': 'with respect to',

    # Frequently occurring masked tokens
    'xxxx': 'xxxx',
    'xx': 'xx'
}

# # Chat abbreviations
# chat_words = {
#     'asap': 'as soon as possible',
#     'pls': 'please',
#     'plz': 'please',
#     'u': 'you',
#     'ur': 'your',
#     'btw': 'by the way',
#     'idk': 'i do not know',
#     'imo': 'in my opinion',
#     'thx': 'thanks',
#     'msg': 'message',

#     'acct': 'account',
#     'cc': 'credit card',
#     'amt': 'amount',
#     'txn': 'transaction',
#     'bk': 'bank',
#     'svc': 'service',
#     'cust': 'customer'
# }

#
def preprocess_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Replace chat abbreviations
    words = text.split()

    words = [
        chat_words[word]
        if word in chat_words
        else word
        for word in words
    ]

    text = " ".join(words)

    #Replace emails, URLs, and phone numbers with placeholders
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)

    text = re.sub(r'http\S+|www\S+', ' URL ', text)

    text = re.sub(r'\b\d{10,}\b', ' PHONE ', text)

    # Remove punctuation
    text = re.sub(
        r'[^\w\s]',
        '',
        text
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    # # Remove stopwords
    # words = text.split()

    # words = [
    #     word
    #     for word in words
    #     if word not in stop_words
    # ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


