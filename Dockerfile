FROM python:3.11-slim-bookworm

ARG APT_DEBIAN_MIRROR=http://deb.debian.org/debian
ARG APT_SECURITY_MIRROR=http://deb.debian.org/debian-security
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=

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

# Build/runtime libs for the optional C++ backend and scientific wheels.
RUN set -eux; \
    sed -i \
        -e "s|http://deb.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian|${APT_DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libfreetype6 \
        libgomp1 \
        libpng16-16 \
        ninja-build; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ -n "${PIP_TRUSTED_HOST}" ]; then \
        python -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"; \
        python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" -r /app/requirements.txt; \
    else \
        python -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}"; \
        python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /app/requirements.txt; \
    fi

COPY . /app
RUN python -m pip install --no-deps -e /app

ENTRYPOINT ["python", "-m", "experiments.run_experiments"]
CMD ["--output-dir", "experiments/outputs"]
