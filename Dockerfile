FROM python:3.12-slim

LABEL authors="dhhaics2004@gmail.com"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
