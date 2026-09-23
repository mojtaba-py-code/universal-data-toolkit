# Multi-stage build: the wheel is built once and installed into a slim runtime
# image that carries no build tools.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m build --wheel --outdir /dist


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="universal-data-toolkit" \
      org.opencontainers.image.description="Ingest, validate, clean, transform and export tabular data" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libpq is only needed when the postgres extra is installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# The wheel name is resolved into a variable: `pip install /tmp/*.whl[postgres]`
# does not work, because the shell reads `[postgres]` as a glob character class.
COPY --from=builder /dist /tmp/dist
RUN wheel="$(ls /tmp/dist/*.whl)" \
    && pip install --no-cache-dir "${wheel}[postgres]" \
    && rm -rf /tmp/dist

# The container runs as an unprivileged user; /data is the mount point for the
# host's files so nothing is written into the image.
RUN useradd --create-home --uid 10001 datatool
WORKDIR /data
RUN chown datatool:datatool /data
USER datatool

ENTRYPOINT ["data-tool"]
CMD ["--help"]
