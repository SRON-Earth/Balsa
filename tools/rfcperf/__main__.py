import sys

from .app import main

try:
    sys.exit(main())
except Exception as exception:
    message = str(exception)
    if message:
        if not message.endswith("."): message += "."
        print("\033[31m" + "ERROR: " + message + "\033[0m")
    else:
        print("\033[31m" + "ERROR: Internal error." + "\033[0m")
        raise
    sys.exit(1)
