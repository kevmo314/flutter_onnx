"""Generate fixtures for RTChat's three-segment playback integration test."""

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("supertonic_reference", sys.argv[1])
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
segments = [
    "Daisy said,",
    "that was the cleanest clutch I have seen all week,",
    "GG everyone!",
]
reference.CASES = [
    dict(name=f"segment_{i}", text=text, lang="en", voice="M2", speed=1.0, steps=5, seed=i)
    for i, text in enumerate(segments)
]
reference.main(sys.argv[2], sys.argv[3])
