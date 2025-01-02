FROM python:3.9-slim

WORKDIR /app

COPY ../../ch_4/backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ../../ch_4/backend .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]