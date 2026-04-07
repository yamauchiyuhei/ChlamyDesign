FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --prefix=/install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

FROM python:3.11-slim

# Install ViennaRNA from pip
RUN pip install --no-cache-dir ViennaRNA

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/api/ /app/src/api/
COPY src/model/ /app/src/model/
COPY src/evaluation/ /app/src/evaluation/
COPY data/processed/green_algae_chloroplast_cds.csv /app/data/processed/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
