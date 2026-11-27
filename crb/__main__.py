"""`python -m crb` is the same as the `crb` command."""
import sys

from crb.cli import main

if __name__ == "__main__":
    sys.exit(main())
