FROM python:3.12-slim

WORKDIR /app

# Copy flask app contents
COPY flask_app/ /app/

# Install dependencies
RUN pip install --upgrade pip setuptools
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000


# CMD ["python", "app.py"]
# Production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]