# Rust toolchain comes from the canonical rust image; the agent builds a Cargo project.
FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS rust

# The verifier runs under pytest, so the final image is the canonical Python base with
# the Rust toolchain copied in.
FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tmux \
        asciinema \
        gcc \
        g++ \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes --no-deps --no-cache-dir -r /tmp/requirements.lock

# The database export the model is built from, and the Cargo project that consumes it.
COPY data /app/data
COPY predictor /app/predictor

# Warm the registry and compile the dependency tree now, while the build still has a
# network. The sandbox itself is offline, so the agent's builds resolve from this cache
# and only recompile the crate's own sources.
RUN cd /app/predictor \
    && CARGO_NET_OFFLINE=false cargo fetch --locked \
    && cargo build --release --offline \
    && rm -f /app/predictor/target/release/fleet_predictor

WORKDIR /app
