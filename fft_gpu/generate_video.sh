#!/bin/bash
cargo run --release -- --generate-video | ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size 512x256 -framerate 30 -i - -c:v libx264 -pix_fmt yuv420p output.mp4
