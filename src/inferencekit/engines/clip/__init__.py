"""Video clip engines behind the clip HTTP contract, and a ComfyUI client.

  server.py         POST /v1/clips and GET /health around a pipeline object; runs in the pipeline's own
                    virtual environment (standard library plus whatever the pipeline imports).
  noise.py          the initial-noise recipe: drawn on the CPU from one seed, so every engine starts from
                    the same tensors.
  diffusers_ltx.py  LTX-2.x through Diffusers (text-to-video and image-to-video), BF16 reference mode.
  official_ltx.py   the official LTX-2 pipelines package, when it is installed (guarded import).
  comfyui.py        a client for a ComfyUI server (POST /prompt with a pinned API-format workflow), and the
                    ``comfyui`` backend bench clip drives.

Nothing here is imported by the rest of inferencekit unless a clip command asks for it.
"""
