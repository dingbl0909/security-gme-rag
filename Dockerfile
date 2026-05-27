FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p data/db data/uploads

EXPOSE 8020

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8020"]
