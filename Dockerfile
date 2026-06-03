FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY config/ ./config/
COPY src/ ./src/
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.services.app:app", "--host", "0.0.0.0", "--port", "8080"]
