FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
ENV FLASK_APP=app.py

EXPOSE 7860

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
