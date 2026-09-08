FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Pre-satisfy torch+torchvision with matching CPU-only wheels from the same
# index *before* the main resolve. docling's tree pulls both transitively
# (via docling-ibm-models / transformers) with no index override, so pip/uv
# would otherwise also have to solve their versions -- and the default
# index resolves the much larger CUDA wheels -- simultaneously with
# everything else, which is the single biggest contributor to multi-minute
# backtracking observed installing docling. Both must come from this same
# install so their compiled ABIs actually match -- installing torch alone
# here let uv pick a torchvision from PyPI's default index afterwards that
# wasn't built against this exact torch build, breaking at import time with
# "operator torchvision::nms does not exist".
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# uv's resolver is substantially faster than pip's and reports conflicts
# immediately instead of silently backtracking for a long time.
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
