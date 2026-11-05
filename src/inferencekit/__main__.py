"""Allow ``python -m inferencekit ...`` as an alias of the ``inferencekit`` command."""

from inferencekit.cli import main

raise SystemExit(main())
