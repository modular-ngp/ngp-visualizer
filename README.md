# ngp-visualizer

This project builds on top of the Vulkan Visualizer template to provide a dense neural radiance field point cloud debugger. It renders synthetic data out of the box and can subscribe to Instant NGP training streams via the bundled SHMX client (shmx_client.h).

## Highlights

- Vulkan 1.3 renderer with ImGui docking UI.
- Live point-cloud controls (density windowing, colour mapping, playback tweaks).
- SHMX shared-memory client that ingests data from the Python NGPDebugServer packaged with ngp-baseline-torch.

## Building

```
cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release
```

## Streaming Data

Run the Python debug server from `ngp-baseline-torch` (for example `example_debug_server.py`). The visualizer connects to the `ngp_debug` shared-memory stream by default; you can change the stream name from the Field Controls panel.