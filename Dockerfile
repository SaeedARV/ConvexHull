FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    USER=app \
    LOGNAME=app

WORKDIR /app

# Runtime libs for common scientific/python wheels (matplotlib Agg, numba, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 \
        libpng16-16 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENTRYPOINT ["python", "-m", "experiments.run_experiments"]
CMD ["--output-dir", "experiments/outputs"]
