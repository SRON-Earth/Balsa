import sys

from .app import main

try:
    sys.exit(main())
except Exception as exception:
    message = str(exception)
    if message:
        if not message.endswith("."): message += "."
        print("ERROR: " + message, file=sys.stderr)
    else:
        raise
    sys.exit(1)
