# syntax=docker/dockerfile:1
#
# Multi-stage build — no `apt-get` anywhere.
#   * the uv base images already ship Python 3.14 + uv;
#   * the non-slim `bookworm` tag (used by `deps`/`dev`) also ships
#     git / curl / ca-certificates, so the devcontainer needs no extra
#     OS packages installed by hand;
#   * the `runtime` stage starts from the smaller `-slim` tag and only
#     copies the pre-built virtualenv across.
#
# Targets:
#   dev      → built by .devcontainer/devcontainer.json
#   runtime  → `docker build -t embeddingclient .`  (default, last stage)

# --------------------------------------------------------------------------
# deps — resolve the locked dependency set into /opt/venv
# --------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm AS deps

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only pyproject.toml + uv.lock are needed to build the venv — bind-mounting
# them (instead of COPY) keeps this layer cached until the lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

# --------------------------------------------------------------------------
# dev — devcontainer target. Inherits the venv from `deps`; the non-slim
#       base already provides git/curl, so no apt-get is required.
# --------------------------------------------------------------------------
FROM deps AS dev

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
CMD ["bash"]

# --------------------------------------------------------------------------
# runtime — slim image that runs the model end to end
# --------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY pyproject.toml uv.lock ./
COPY src ./src

# Generate the synthetic dataset, then run the MTM + InfoNCE pre-training.
CMD ["sh", "-c", "python -m src.make_dataset && python -m src.train"]
