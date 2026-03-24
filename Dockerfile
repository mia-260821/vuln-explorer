FROM python:3.11-slim-bookworm


ENV APP_HOME=/app

WORKDIR ${APP_HOME}

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends build-essential curl \
#     && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir ${APP_HOME} appuser

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install fastapi uvicorn[standard]

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY ingest-entrypoint.sh /usr/local/bin/ingest-entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/ingest-entrypoint.sh

COPY . ${APP_HOME}

RUN chown -R appuser:appuser ${APP_HOME}

USER appuser

EXPOSE 8501 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
