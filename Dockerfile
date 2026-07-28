FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        "graphreduce==1.9.17" \
        "relbench==2.1.1" \
        "catboost==1.2.10" \
        "duckdb==1.2.2" \
        "numpy<2" \
        "pandas>=1.5,<3" \
        "pyarrow==23.0.1" \
        "scikit-learn==1.6.0" \
        "pooch>=1.8,<2" \
        "pytest>=8,<9"

ENTRYPOINT ["python", "scripts/run_all.py"]
