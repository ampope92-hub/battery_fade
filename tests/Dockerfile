# Verifier image. Runs in its own container: it never sees the agent's filesystem,
# only the paths declared in `artifacts`. It rebuilds the agent's crate from source and
# grades the predictions that come out, so it needs the Rust toolchain as well as pytest.

FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS rust

FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup

# Verifier tooling, pinned. Nothing is installed at trial time.
RUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5

# The crate skeleton mirrors environment/predictor's manifest so the dependency tree can
# be fetched and compiled now, while the build still has a network. The agent's own
# src/, Cargo.toml and Cargo.lock arrive later as artifacts and overwrite these, leaving
# only the crate's own sources to recompile. Keep this in sync with
# environment/predictor/Cargo.{toml,lock}.
COPY skeleton /app/predictor
RUN cd /app/predictor \
    && CARGO_NET_OFFLINE=false cargo fetch --locked \
    && cargo build --release --offline \
    && rm -rf /app/predictor/src

# Artifact landing directories must exist before Harbor uploads the declared artifacts.
RUN mkdir -p /app/predictor/src /app/data

COPY . /tests/

WORKDIR /app
