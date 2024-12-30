# Video Resolution with GFPGAN

This project applies GFPGAN super-resolution to video frames, specifically focusing on enhancing facial regions in videos.

## Prerequisites

- Python 3.8 or higher
- CUDA capable GPU (recommended)
- FFmpeg installed on your system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VishnuH28/Video-quality-enhancer.git
cd Video-quality-enhancer
```

2. Script to run:
```bash
python x.py --superres GFPGAN -iv input_video.mp4 -ia input_audio.mp3 -o output.mp4
