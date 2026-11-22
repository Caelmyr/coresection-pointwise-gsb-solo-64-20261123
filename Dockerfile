ARG CORESECTION_DEPS_IMAGE=python:3.11-slim
FROM ${CORESECTION_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/coresection
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 CORESECTION_DATA=/var/lib/coresection
WORKDIR /opt/coresection
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY coresection ./coresection
COPY static ./static
RUN useradd -U -u 11517 coresection && install -d -o coresection -g coresection /var/lib/coresection
USER coresection
EXPOSE 8000
CMD ["python", "-m", "coresection"]
