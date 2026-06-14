# Headless container for the path-planning pipeline.
#
# Builds and runs on both x86_64 (a laptop) and aarch64 (the Jetson): numpy and
# scipy ship prebuilt wheels for both, so nothing is compiled from source.
# It runs the simulation only -- no pygame, no display -- which is exactly what
# a Jetson running locally needs.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime that the BLAS bundled inside numpy/scipy needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first so this layer stays cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# By default, drive every mission end to end as a smoke test (exits non-zero if
# any mission fails to finish). Override the command to do something else, e.g.
#   docker run --rm path-planning python run_headless.py autocross
#   docker run --rm path-planning python -m unittest -v
CMD ["python", "run_headless.py", "all"]
