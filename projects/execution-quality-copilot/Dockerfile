FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Copy the rest and build the synthetic dataset into the image.
COPY config.yaml ./
RUN python -m execution_quality_copilot.datagen

# MCP servers speak JSON-RPC over stdio; run the server as the entrypoint.
ENTRYPOINT ["python", "-m", "execution_quality_copilot.server.main"]
