# One container: builds the React dashboard, then serves it and the API from FastAPI.
#
#   docker build -t frection .
#   docker run -p 8000:8000 frection          ->  http://localhost:8000
#
# WITH_GNN=1 (default) installs CPU PyTorch so the GraphSAGE risk score is available.
# Build with --build-arg WITH_GNN=0 for small hosts (about 512 MB RAM); the rules,
# customer-records mode and explanations all work without it.

FROM node:20-slim AS frontend
WORKDIR /frontend
COPY dashboard/frontend/package*.json ./
RUN npm ci
COPY dashboard/frontend/ ./
RUN npm run build

FROM python:3.12-slim
ARG WITH_GNN=1
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements-app.txt ./
RUN pip install -r requirements-app.txt
RUN if [ "$WITH_GNN" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install torch_geometric; \
    fi

COPY api/ api/
COPY src/ src/
COPY models/ models/
COPY --from=frontend /frontend/dist dashboard/frontend/dist

# Hosts such as Render and Hugging Face Spaces provide the port in $PORT
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
