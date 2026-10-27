FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 REGLINGUA_DEMO_MODE=1 REGLINGUA_LLM_ENABLED=0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Retrain classifiers with the sklearn pinned in this image and build the offline bridge.
RUN python scripts/bootstrap_demo.py
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
