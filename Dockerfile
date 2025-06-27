FROM python:3.9-slim

WORKDIR /app

# Only copy what's essential
COPY application.py .
COPY requirements.txt .
COPY templates/ ./templates/
COPY artifacts/model.pkl ./artifacts/model.pkl

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "application.py"]
