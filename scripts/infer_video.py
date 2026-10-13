"""Six-panel CausticsNet and BackscatterNet inference on a video.

    uv run python scripts/infer_video.py VIDEO -o panels.mp4 \\
        --caustics runs/causticsnet/checkpoints/latest.pt \\
        --backscatter runs/backscatternet/checkpoints/latest.pt

Caustics, backscatter parameters and depth are predicted from the original RGB
frame. The backscatter veil is subtracted from the caustics-cleaned frame, so
the third panel has both effects removed. See ``caustics_backscatter.infer``.
"""

from caustics_backscatter.infer import main

if __name__ == "__main__":
    main()
