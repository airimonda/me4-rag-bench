#!/usr/bin/env bash
# Install crb on a 64-bit Raspberry Pi OS. Safe to run again.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEED_MB=2500

say() { printf '%s\n' "$*"; }
die() { printf 'Problem: %s\n' "$*" >&2; exit 1; }

say "Installing crb from $REPO"

# 1. 64-bit OS
arch="$(uname -m)"
case "$arch" in
  aarch64|arm64|x86_64) ;;
  *) say "Note: this is a $arch system. crb needs 64-bit Raspberry Pi OS. Continuing anyway." ;;
esac

# 2. Python 3.10 or newer
command -v python3 >/dev/null 2>&1 || die "python3 not found. Run: sudo apt install python3"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  die "Python is $(python3 -V 2>&1) but 3.10 or newer is needed. Use Raspberry Pi OS Bookworm (64-bit) or newer."
fi
say "Python ok: $(python3 -V 2>&1)"

# 3. venv support
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  say "python3-venv is missing. Installing it with apt (it will ask for your password)."
  sudo apt-get update
  sudo apt-get install -y python3-venv
fi

# 4. free disk
free_mb="$(df -Pm "$REPO" | awk 'NR==2 {print $4}')"
if [ "${free_mb:-0}" -lt "$NEED_MB" ]; then
  die "Only ${free_mb} MB free here, crb needs about ${NEED_MB} MB. Free some space (docs/TROUBLESHOOTING.md, 'Disk full') and run this again."
fi
say "Disk ok: ${free_mb} MB free"

# 5. venv + install
if [ ! -x "$REPO/.venv/bin/python" ]; then
  say "Creating the virtual environment in $REPO/.venv"
  python3 -m venv "$REPO/.venv"
else
  say "Virtual environment already exists, reusing it"
fi
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip
say "Installing crb (this can take a few minutes on a Pi)"
"$REPO/.venv/bin/python" -m pip install --quiet -e "$REPO[build]"

"$REPO/.venv/bin/crb" --help >/dev/null 2>&1 || die "crb installed but does not start. Re-run this script and send the output to the teaching team."

if ! command -v vcgencmd >/dev/null 2>&1; then
  say "Note: vcgencmd not found. The report will have no temperature or throttling numbers."
fi

cat <<EOF

Done. Next steps:

  1. Activate the environment (every new terminal):
       cd $REPO
       source .venv/bin/activate

  2. Download the data (about 1.2 GB, one time):
       crb fetch

  3. Start your adapter in a second terminal (or try the reference one: crb baseline).
     Then set your key:
       export OPENROUTER_API_KEY=...

  4. Run the benchmark:
       crb

Full guide: docs/GUIDE.md
EOF
