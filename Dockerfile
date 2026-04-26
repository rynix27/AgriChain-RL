FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py .
COPY agrichain.py .
COPY environment.py .
COPY mandi_env.py .
COPY models.py .
COPY whatsapp_alerts.py .
COPY inference.py .
COPY example_agent.py .
COPY server/ ./server/

EXPOSE 7860

CMD ["python", "server/app.py"]
