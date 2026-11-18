"""Allow ``python -m bhagavatam_ai ...`` as an alias for the ``bhagavatam`` CLI."""

from bhagavatam_ai.cli import main

raise SystemExit(main())
