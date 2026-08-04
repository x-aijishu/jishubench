# Development Guide

## 1. Install rustup and the stable toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
rustup component add rustfmt clippy
```

Verify:

```bash
rustc --version    # recommended: >= 1.75
cargo --version
```

## 2. IDE setup

Install the **rust-analyzer** extension in VS Code / Cursor, then open
`jishubench/target/` as the workspace root so rust-analyzer picks up the
Cargo workspace correctly.

## 3. Build and run on the local machine

```bash
cd jishubench/target
cargo build                                         # debug, OS metrics only
cargo build --features nvidia                       # debug, with NVIDIA GPU monitoring
cargo build --features plugin                       # debug, with .so profiler plugin support
cargo build --features nvidia,plugin                # debug, all features
cargo build --release                               # release (OS metrics only)
cargo run -p agent -- --bind 127.0.0.1:9090         # OS metrics only
cargo run -p agent --features nvidia -- --bind 127.0.0.1:9090  # with GPU monitoring
```

Verify the agent is up:

```bash
curl http://127.0.0.1:9090/v1/eval/health
# or from the repo root:
jishubench preflight    # should show monitor_reachable: true
```

## 4. Cross-compile for Linux (deployment)

Pick the target triple that matches your board or server. Rust itself can be
installed via rustup on any platform -- only the **cross-linker** (C compiler
for the target) is platform-specific.

!!! tip "Native Linux x86_64"
    Skip cross-compilation and run `cargo build --release` -- the output is
    `target/release/edge-eval-agent`. The linker configuration below is only
    needed when cross-compiling from a different host OS or architecture.

### Configure the cross-linker (multi-platform)

The linker binary name differs between macOS (Homebrew messense) and Linux
(apt). Do **not** commit per-machine linker settings to the repo -- configure
them once on each development machine instead.

Add the matching block to `~/.zshrc`, `~/.bashrc`, or your shell profile:

```bash
# Linux (apt cross toolchains)
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc

# macOS (Homebrew messense) -- use this block instead on macOS
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-unknown-linux-gnu-gcc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc
```

### Target arch: aarch64 (ARM board)

**Step 1 -- Add the Rust target:**

```bash
rustup target add aarch64-unknown-linux-gnu
```

**Step 2 -- Install the cross-linker:**

```bash
# macOS (Homebrew)
brew install messense/macos-cross-toolchains/aarch64-unknown-linux-gnu

# Ubuntu / Debian
sudo apt install gcc-aarch64-linux-gnu
```

**Step 3 -- Build:**

```bash
cd jishubench/target
cargo build --release --target aarch64-unknown-linux-gnu
# Output: target/aarch64-unknown-linux-gnu/release/edge-eval-agent
# With NVIDIA GPU support:
cargo build --release --features nvidia --target aarch64-unknown-linux-gnu
```

### Target arch: x86_64 (Linux PC / server)

**Step 1 -- Add the Rust target:**

```bash
rustup target add x86_64-unknown-linux-gnu
```

**Step 2 -- Install the cross-linker:**

```bash
# macOS (Homebrew)
brew install messense/macos-cross-toolchains/x86_64-unknown-linux-gnu

# Ubuntu / Debian
sudo apt install gcc-x86-64-linux-gnu
```

**Step 3 -- Build:**

```bash
cd jishubench/target
cargo build --release --target x86_64-unknown-linux-gnu
# Output: target/x86_64-unknown-linux-gnu/release/edge-eval-agent
# With NVIDIA GPU support:
cargo build --release --features nvidia --target x86_64-unknown-linux-gnu
```

## 5. Deploy and run on the target

```bash
# First-time setup: create ~/.jishubench/bin on the target
ssh user@board 'mkdir -p ~/.jishubench/bin'

# Copy the binary -- pick the path that matches your build target:
scp target/aarch64-unknown-linux-gnu/release/edge-eval-agent \
    user@board:~/.jishubench/bin/edge-eval-agent
# or:
# scp target/x86_64-unknown-linux-gnu/release/edge-eval-agent \
#     user@board:~/.jishubench/bin/edge-eval-agent

# Make it executable
ssh user@board 'chmod +x ~/.jishubench/bin/edge-eval-agent'

# Start the agent
ssh user@board \
    '~/.jishubench/bin/edge-eval-agent --bind 0.0.0.0:9090'
```

To run `edge-eval-agent` without the full path, add `~/.jishubench/bin` to
your PATH on the board:

```bash
echo 'export PATH="$HOME/.jishubench/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc
source ~/.bashrc
edge-eval-agent --bind 0.0.0.0:9090
```

!!! tip "Building directly on the board"
    If the board already has Rust installed, you can build directly on the
    board with `cargo build --release` -- no cross-compilation needed. Copy
    the resulting binary with:
    `cp target/release/edge-eval-agent ~/.jishubench/bin/edge-eval-agent`

!!! tip "NVIDIA GPU monitoring"
    To enable GPU metrics collection, pass `--features nvidia` at build time.
    The resulting binary will probe for NVML (`libnvidia-ml.so`) at runtime
    and gracefully degrade to `None` fields if no GPU is found. This is safe
    to enable on non-NVIDIA hardware.

## 6. Runtime requirements on the board

The binary links dynamically against glibc by default. Confirm compatibility:

```bash
ldd ~/.jishubench/bin/edge-eval-agent
```

For boards with very old glibc, we can switch to `aarch64-unknown-linux-musl`
for a fully static binary -- the current MVP uses the `gnu` target.
