"""inferencekit: measure-first LLM inference optimisation.

Every operation produces an *evidence record*: a small JSON document that says what
was measured, on which GPU / engine / model / workload, with which tool, and what
the result does NOT show (``limitations``). Records are content-addressed, so a
report can cite exactly the numbers it used.
"""

__version__ = "0.1.0"
