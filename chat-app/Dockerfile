FROM python:3.12-slim

WORKDIR /app
COPY server.py .
COPY static ./static

ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 3000
CMD ["python", "server.py"]
