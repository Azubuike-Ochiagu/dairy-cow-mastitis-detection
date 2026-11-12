FROM python:3.14.2-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py model.bin model_metadata.json ./

EXPOSE 9696

CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]