FROM python:3.9-slim

WORKDIR /app

COPY ../../ch_4/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../../ch_4/frontend .

CMD ["streamlit", "run", "app.py"]