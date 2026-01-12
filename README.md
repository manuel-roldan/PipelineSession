# PipelineSession (C++17)

A C++17 framework for building, validating, running, and debugging GStreamer pipelines with a typed, composable API. It focuses on deterministic pipeline generation, strong diagnostics, and ML-friendly outputs.

## Highlights
- Typed Nodes and NodeGroups that generate deterministic GStreamer fragments.
- `PipelineSession` runtime with validation, structured diagnostics, and safe teardown.
- Debugging via `DebugPoint`, `run_tap()`, and `run_debug()` with per-tap outputs.
- Input groups for RTSP, video, and images (auto JPEG/PNG decode).
- Tensor-friendly output path with `TensorStream` and `FrameTensorRef`.
- Pipeline save/load (JSON) for reproducibility and tooling.
- Human-friendly views with `describe()` and `to_gst()`.

## Build
Requirements:
- GStreamer (core, app, video, rtsp-server)
- OpenCV (currently required by CMake)

Build:
```bash
cmake -S . -B build
cmake --build build
```

Run tests:
```bash
ctest --test-dir build
```

## Quick example
```cpp
sima::PipelineSession p;

sima::nodes::groups::ImageInputGroupOptions in;
in.path = "test.jpg";
in.output_caps.width = 256;
in.output_caps.height = 256;
p.add(sima::nodes::groups::ImageInputGroup(in));

sima::OutputTensorOptions out;
out.format = "RGB";
out.target_width = 256;
out.target_height = 256;
p.add_output_tensor(out);

sima::TensorStream ts = p.run_tensor();
auto frame = ts.next(/*timeout_ms=*/2000);
```

## Status
This repository is under active development. Python bindings are not included yet; the C++ API provides a zero-copy-friendly tensor interface to support future bindings.

## Docs
See `docs/ARCHITECTURE.md` for the detailed design, runtime model, and module responsibilities.
