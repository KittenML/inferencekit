"""Every operation module registers itself on import. Order = order in ``inferencekit --help``."""

from inferencekit.ops import (  # noqa: F401
    doctor,
    roofline,
    ceiling,
    bench,
    nsys,
    ncu,
    rocprof,
    compare,
    correctness,
    store,
    report,
    review,
    ledger,
    manifest,
)
from inferencekit.ops import intake  # noqa: F401,E402
from inferencekit.ops import resume  # noqa: F401,E402
from inferencekit.ops import quality  # noqa: F401,E402
from inferencekit.ops import bench_serve  # noqa: E402,F401
from inferencekit.ops import bench_rounds  # noqa: F401,E402
from inferencekit.ops import prompts  # noqa: F401,E402
from inferencekit.ops import gates  # noqa: F401,E402
from inferencekit.ops import bench_clip  # noqa: F401,E402  (video: bench clip)
from inferencekit.ops import quality_clip  # noqa: F401,E402  (video: quality clip)
from inferencekit.ops import quality_clip_steps  # noqa: F401,E402  (video: quality clip-steps)
from inferencekit.ops import gates_clip  # noqa: F401,E402  (video: gates clip; needs bench_clip)
from inferencekit.ops import bench_image  # noqa: F401,E402
from inferencekit.ops import bench_audio  # noqa: F401,E402
from inferencekit.ops import quality_audio  # noqa: F401,E402
from inferencekit.ops import cpu  # noqa: F401,E402
