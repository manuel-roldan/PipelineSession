# PipelineSession (C++17)

A C++17 framework for building, validating, running, and debugging GStreamer pipelines with a typed, composable API. It focuses on deterministic pipeline generation, strong diagnostics, and ML-friendly outputs.

## Highlights
- Typed Nodes and NodeGroups that generate deterministic GStreamer fragments.
- `PipelineSession` runtime with validation, structured diagnostics, and safe teardown.
- Debugging via `DebugPoint`, `run_tap()`, and `run_debug()` with per-tap outputs.
- Input groups for RTSP, video, and images (auto JPEG/PNG decode).
- MPK integration via `ModelMPK` and a simple `ModelSession` wrapper.
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

Note: MPK tests will download assets if missing (ResNet50 MPK + ImageNet goldfish image).
Set `SIMA_RESNET50_TAR` to a local MPK tarball or run `sima-cli modelzoo get resnet_50`
to avoid repeated downloads.

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

## MPK / ModelSession
```cpp
std::vector<float> mean = {0.485f, 0.456f, 0.406f};
std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

simaai::ModelSession session;
if (!session.init("tmp/resnet_50_mpk.tar.gz",
                  224, 224, "RGB",
                  /*normalize=*/true,
                  mean, stddev)) {
  throw std::runtime_error(session.last_error());
}

cv::Mat rgb = /* load or convert to RGB */;
auto out = session.run_tensor(rgb);
```

Note: SimaAI plugins are single-owner and single-use per process. If you need to run
multiple SimaAI pipelines, spawn a child process for each run.

## Status
This repository is under active development. Python bindings are not included yet; the C++ API provides a zero-copy-friendly tensor interface to support future bindings.

## Docs
See `docs/ARCHITECTURE.md` for the detailed design, runtime model, and module responsibilities.
