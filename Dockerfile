FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY model.keras /app/model.keras
COPY scaler.save /app/scaler.save

EXPOSE 8000

CMD ["uvicorn", "src.main_docker:app", "--host", "0.0.0.0", "--port", "8000"]
