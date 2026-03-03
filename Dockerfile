FROM rust:1.84-slim-bookworm

WORKDIR /app

# Install pkg-config and ALSA dev headers (needed by some audio crates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN cargo build --workspace 2>&1

CMD ["cargo", "test", "--workspace"]
