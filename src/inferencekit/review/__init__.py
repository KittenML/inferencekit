"""The review page: every engine's outputs side by side, per input bucket, in one offline HTML file.

``rundir`` reads ``runs/<run-id>/``; ``diff`` compares texts (first divergence, word
diffs, word errors); ``page`` renders the HTML. The ``review build`` operation
(``inferencekit.ops.review``) ties them together.
"""
