FROM bitnami/spark:latest

ENV HOME=/app
ENV IVY_HOME=/tmp/.ivy2

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src
COPY data/ ./data
COPY logs/ ./logs

CMD ["tail", "-f", "/dev/null"]
