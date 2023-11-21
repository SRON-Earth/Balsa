import sys

from .app import main

try:
    sys.exit(main())
except Exception as exception:
    print(f"ERROR: '{exception}'.", file=sys.stderr)
    sys.exit(1)