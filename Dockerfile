FROM python:3.11-slim

WORKDIR /artifact
COPY . .
RUN python3 -m pip install --no-cache-dir .

CMD ["python3", "scripts/verify_artifact.py"]
