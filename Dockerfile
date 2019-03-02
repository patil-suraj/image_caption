FROM python:2.7-slim-stretch

RUN apt-get update && apt-get install -y git python2.7 gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt --upgrade

COPY app app/

RUN python app/server.py

EXPOSE 8080

CMD ["python", "app/server.py"]