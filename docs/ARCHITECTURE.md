# Repository Architecture & Design

This repository is a **C++17 library for building, validating, running, and debugging GStreamer pipelines** using a stable, composable API. It provides a higher-level “pipeline-as-code” interface while keeping GStreamer’s power and flexibility, with ML-friendly outputs (tensors) and reproducible configs.

The guiding idea is:

> **Typed, deterministic building blocks (Nodes) → a reproducible GStreamer launch string → a managed runtime (PipelineSession) with strong diagnostics (PipelineReport).**

---

## What this library is for

### Primary users
Developers who want to:
- Assemble pipelines from reusable building blocks (without writing raw GStreamer boilerplate)
- Validate pipelines early (CI-friendly) and understand failures quickly
- Run pipelines and consume frames in C++ via `appsink`
- Tap/debug intermediate points in a pipeline (including per-tap tensor snapshots)
- Optionally serve a pipeline over RTSP (via `gst-rtsp-server`)
- Feed ML code via tensor-friendly outputs without writing GStreamer plumbing

### Common workflows
- **Decode / ingest:** file or RTSP → depay/demux/parse → decode → convert/caps → appsink → C++ consumer
- **Tap/debug:** insert a `DebugPoint("X")`, then `run_tap("X")` to inspect bytes / frames at that boundary
- **Validate:** build + parse + preroll (PAUSED) to catch negotiation issues early
- **Serve RTSP:** push synthetic frames into an RTSP server pipeline using `appsrc`
- **ML output:** image/video/RTSP → decode → convert/scale → `add_output_tensor(...)` → `TensorStream`

---

## Repository layout

### High-level structure
- `include/` — public headers (the supported API surface)
- `src/` — implementations
- `docs/` — design docs (this file)
- `examples/` — small runnable examples
- `tests/` — unit/integration tests
- `old_PipelineSession.*` — legacy monolithic implementation kept for reference/migration

### Namespaced headers (`include/` vs `include/sima/`)
You will see two parallel header trees:

- `include/<module>/...`
- `include/sima/<module>/...`

The `include/sima/...` headers are **include shims** to provide a stable top-level include path (e.g. `#include "sima/pipeline/PipelineSession.h"`). The “real” headers are in `include/pipeline`, `include/gst`, etc. The shim headers should remain thin and free of implementation logic.

---

## Modules and responsibilities

### `builder/` — graph & composition (no GStreamer)
**Purpose:** Define how pipelines are assembled from logical parts.

Key types:
- `Node` — interface implemented by each pipeline building block
- `Graph`, `Builder`, `NodeGroup` — composition utilities and printing

**Rule:** builder must remain mostly STL-only. It should not own GStreamer runtime objects.

---

### `nodes/` — typed pipeline building blocks
**Purpose:** Provide ready-to-use Node implementations that emit deterministic GStreamer fragments.

Examples:
- `nodes/io/RTSPInput`, `nodes/io/AppSrcImage`
- `nodes/common/*` (Caps, Queue, DebugPoint, AppSink, etc.)
- `nodes/sima/*` (SiMa decode/encode/parse/pay nodes)
- `nodes/rtp/*` (depay/payload helpers)
- `nodes/groups/*` (common multi-node recipes)

**Contract:**
Each Node must produce:
- `gst_fragment(index)` — the GStreamer fragment for this node at a given index
- `element_names(index)` — deterministic element names owned by this node (for diagnostics and enforcement)

---

### `gst/` — thin GStreamer utilities
**Purpose:** Small wrappers/helpers around common GStreamer patterns.

Examples:
- initialization (`GstInit`)
- parsing launch strings (`GstParseLaunch`)
- bus draining/stringifying (`GstBusWatch`)
- caps helpers / element introspection (`GstHelpers`, `GstIntrospection`)
- pad taps / probe helpers (`GstPadTap`)

**Rule:** `gst/` must not depend on `pipeline/` (to avoid dependency cycles and “utility layer” bloat).

---

### `pipeline/` — runtime orchestration and public API
**Purpose:** Own the runtime lifecycle: build → parse → run → consume → teardown, with diagnostics.

Key types:
- `PipelineSession` — the main entry point for users
- `FrameStream` — consume frames from `appsink` (strict expectations)
- `TensorStream` — consume tensors from `appsink` (ML-friendly)
- `TapStream` — debug/tap intermediate points (more permissive)
- `PipelineReport` — structured diagnostics for failures, stalls, and reproduction
- `Errors` — exceptions (`PipelineError`) embedding a report

#### Internal pipeline diagnostics
Under `include/pipeline/internal/`:
- `Diagnostics.h` — shared diagnostics types used by runtime:
  - `DiagCtx` (bus log + node reports + boundary counters)
  - `BoundaryFlowCounters` (atomic counters updated from streaming threads)
- `GstDiagnosticsUtil.h` — helpers for formatting and collecting GStreamer diagnostics

---

### `contracts/` — validation rules
**Purpose:** Encode “what a valid pipeline looks like” beyond “gst_parse_launch succeeded”.

Examples:
- validator interfaces and registries
- structured `ValidationReport`

This layer can be used for CI and for catching issues before runtime.

---

### `policy/` — user-tunable behavior
**Purpose:** Centralize tunables (defaults, memory constraints, encoder/decoder/RTSP policy choices).

The goal is to make “knobs” explicit and discoverable rather than hidden in scattered code.

---

### `mpk/` — MPK integration
**Purpose:** Load/interpret “model packs” (MPK) and adapt them into pipeline nodes or pipeline fragments.

This module is intentionally optional and should not contaminate the core runtime path unless used.

Key types:
- `ModelMPK` — loads an MPK tarball, parses its JSON, and exposes model fragments
- `ModelStage` — `Preprocess`, `MlaOnly`, `Postprocess`, `Full`
- `ModelMPKOptions` — input size/format/normalization and upstream naming
- `ModelFragment` — `{gst, elements}` pair for deterministic fragments

Common usage:

```cpp
sima::mpk::ModelMPKOptions opt;
opt.input_width = 224;
opt.input_height = 224;
opt.input_format = "RGB";
opt.normalize = true;
opt.mean = {0.485f, 0.456f, 0.406f};
opt.stddev = {0.229f, 0.224f, 0.225f};

auto model = sima::mpk::ModelMPK::load("resnet_50_mpk.tar.gz", opt);
sima::PipelineSession p;
p.add(sima::nodes::groups::Infer(model));
```

`ModelMPK::to_node_group(ModelStage)` returns a `NodeGroup` for a specific stage.
The `sima::nodes::groups::{Preprocess,MLA,Postprocess,Infer}` helpers wrap that
API and should be preferred when composing pipelines from an already-loaded model.

`ModelMPK::input_appsrc_options(...)` provides caps/config for `InputAppSrc`
when you need to feed frames or tensors into an MPK pipeline.

---

### `ModelSession` — MPK inference wrapper
**Purpose:** Provide a simple, single-model inference API on top of `ModelMPK` and `PipelineSession`.

Key properties:
- owns a `PipelineSession` configured for a single MPK (full chain)
- takes explicit preproc settings (width/height/format/normalization)
- exposes `run_tensor()` for ML-friendly outputs
- owns a process-level SimaAI guard (see below)
- exposes `last_error()` for failure diagnostics

`init(tar_gz, width, height, format, normalize, mean, stddev)` configures
the EV74 preproc path for accuracy-sensitive models (e.g. ImageNet).

Call `close()` (or rely on destructor) to release the guard and session resources.

---

## Runtime model (how execution works)

### Initialization
All runtime entry points call a single safe initialization routine:
- `gst_init_once()` (thread-safe, `std::call_once`)

Additionally, runtime paths may verify required plugins are present:
- `require_element("appsink", ...)`, etc.

### Building pipelines
A `PipelineSession` is built by adding `Node` objects:

```cpp
sima::PipelineSession s;
s.add(nodes::RTSPInput("rtsp://..."))
 .add(nodes::H264DecodeSima())
 .add(nodes::Caps(/*...NV12...*/))
 .add(nodes::OutputAppSink());
````

Internally:

1. The session asks each Node for `gst_fragment(i)` and concatenates fragments with `!`
2. Optionally inserts **boundary markers** between nodes:

   * `identity name=sima_b<i> silent=true`
3. Builds a `DiagCtx`:

   * `node_reports` for reproducibility
   * `boundaries` as `BoundaryFlowCounters` (atomics)

### Parsing & launch

The library primarily uses:

* `gst_parse_launch(pipeline_string, &err)`

This provides flexibility and debuggability (you can replay the exact string with `gst-launch-1.0`).

### Running

Typical flow (`PipelineSession::run()`):

1. Enforce contracts (e.g., “sink last” for `run()`)
2. Build pipeline string (+ optional boundaries)
3. Parse pipeline
4. Optionally enforce element naming contract
5. Attach optional boundary probes
6. Set pipeline to `PLAYING`
7. Return a `FrameStream` bound to an `appsink`

`run_tap(name)` is similar, but truncates at a named `DebugPoint` and appends a dedicated `appsink` for the tap.

### Teardown

Teardown is intentionally defensive.
Some plugin stacks can hang on state changes; the runtime prefers to avoid deadlocking the host process/CI.

The common pattern is:

* send EOS
* set `GST_STATE_NULL`
* unref objects
* apply a timeout safeguard (leak instead of hanging if necessary)

---

## SimaAI guard (single-owner + single-use)

SimaAI plugins are not safe to run concurrently or repeatedly in the same process.
To avoid crashes, the runtime enforces a process-level guard:

- **Single-owner:** while a SimaAI pipeline is active, any new attempt returns:
  “SimaAI pipelines are single-owner per process; spawn a child process to run another pipeline.”
- **Single-use:** once a SimaAI pipeline has been run, any future attempts in the same process return:
  “SimaAI pipelines are single-use per process; spawn a child process for subsequent runs.”

Implementation:
- `pipeline/internal/SimaaiGuard` tracks `in_use` and `used_once`.
- `PipelineSession` acquires the guard unless an external guard is supplied via `set_guard()`.
- `ModelSession` acquires the guard at `init()` and holds it for the session lifetime.
- Guard acquisition is conditional on `pipeline_uses_simaai()` detecting simaai elements.

This is a defensive constraint until the simaai plugin lifecycle supports safe re-init.

---

## Threading & ownership model

### Threads

* **GStreamer streaming threads**: pad probes, decoding, scheduling
* **User thread**: `appsink` polling + periodic bus draining
* **RTSP server thread**: GLib main loop for `gst-rtsp-server` mode

### Ownership rules (GStreamer objects)

* GStreamer objects are reference counted.
* If you store a `GstObject*` beyond the scope where it was acquired, you must `gst_object_ref()` it.
* Always `gst_object_unref()` exactly once when done.

### Diagnostics thread safety (important)

Pad probes run on streaming threads, so **diagnostics updated from probes must be lock-free**.

The design is:

* `BoundaryFlowCounters` stores **atomics**
* pad probes only do atomic `fetch_add()` / `store()`
* reporting uses `BoundaryFlowCounters::snapshot()` to convert atomics → `BoundaryFlowStats` (plain ints)

This avoids data races while keeping probes cheap.

---

## Diagnostics & observability

### `DiagCtx` captures:

* the pipeline string (for reproduction)
* node reports (what each node generated)
* bus messages (under a mutex)
* boundary flow counters (atomics)

### Boundary flow probes

When enabled, the runtime attaches pad probes to boundary `identity` elements.
They track:

* buffer counts (in/out)
* last seen PTS (ns)
* last seen wall time (monotonic µs)

This is used to generate “likely stall” summaries:

* “we last saw activity entering/leaving boundary X at T”

### Bus logging and errors

The runtime drains bus messages into `DiagCtx`.
On an error message (`GST_MESSAGE_ERROR`), it throws `PipelineError` including a `PipelineReport` and reproduction hints.

### DOT dumps

If enabled, the runtime can emit DOT graphs via `gst_debug_bin_to_dot_file_with_ts(...)` to a configured directory.

---

## Frame output vs Tap output

### `FrameStream` (strict)

`FrameStream` is meant for “real consumption”, so it tends to enforce strong assumptions (e.g., a CPU-mappable, expected format).

Typical expectations:

* consistent caps at the sink
* predictable memory behavior (often SystemMemory)
* stable format (commonly NV12)

### `TensorStream` (ML-friendly)

`TensorStream` is designed for ML consumers in C++:

* returns `FrameTensorRef` (zero-copy view) or `FrameTensor` (owned copy)
* supports RGB/BGR/GRAY8 and NV12/I420 raw formats
* keeps an internal holder so future Python bindings can adopt without copies
* provides a minimal DLPack-compatible struct for later integration

### `TapStream` (permissive)

`TapStream` is for debugging and inspection:

* tries to map video frames when possible
* falls back to raw buffer mapping when applicable
* can report “not mappable” with reasons (e.g., DMABuf/NVMM)

### run_debug() (per-tap outputs)

`run_debug()` returns:

* `RunDebugTap::packet` (raw bytes + caps)
* `RunDebugTap::tensor` + `last_good_tensor` when mappable
* per-tap error strings while continuing through later segments

This preserves partial results when a later node is misconfigured.

---

## Pipeline serialization (save/load)

Pipelines can be saved and restored as JSON:

* `PipelineSession::save(path)` writes a versioned JSON with node kind/label/fragment/elements
* `PipelineSession::load(path)` rehydrates nodes via a `ConfiguredNode` wrapper

The current schema is intentionally minimal and reproducible, and can evolve to richer
node configs later. This also serves as the bridge for future bindings and tooling.

---

## UX helpers

* `PipelineSession::describe()` uses `GraphPrinter` to render a human-readable node list
* `PipelineSession::to_gst()` returns the gst-launch string for quick debugging

---

## Element naming & determinism

Deterministic element names are a core design principle because they enable:

* `gst_bin_get_by_name()` for sinks and debug points
* stable probe attachment
* stable diagnostics and reproducibility
* optional naming contract enforcement (“every element belongs to some node”)

**Node authors must ensure**:

* fragments include stable `name=` fields when elements must be retrievable
* `element_names()` matches exactly what the fragment creates

---

## Validation & contracts

Validation exists to catch issues earlier than runtime:

* `validate()` can parse and preroll (PAUSED) to detect negotiation stalls
* `contracts/` provides structured validators for “pipeline correctness”

The intended behavior:

* runtime flows throw exceptions on fatal errors
* validation flows return structured reports (CI-friendly)

---

## RTSP server mode

`run_rtsp()` uses `gst-rtsp-server`:

* a server runs in a dedicated thread with a GLib main loop
* on `media-configure`, the code locates the `appsrc` by name and configures caps/properties
* frames are pushed periodically (timer-based) with explicit timestamps

Each client may get its own media instance depending on factory configuration.

---

## Environment / configuration knobs

The runtime supports environment-driven debugging knobs (names may vary by implementation):

* enabling DOT dumps (output directory)
* enabling boundary probes
* enabling naming contract enforcement
* validation preroll timeouts

These knobs are intentionally outside the public API so you can turn them on in CI or in the field without recompiling.

---

## How to extend the library

### Adding a new Node

1. Create a header in `include/nodes/<category>/<YourNode>.h`
2. Implement in `src/nodes/<category>/<YourNode>.cpp`
3. Ensure:

   * `gst_fragment(i)` is valid and deterministic
   * all important elements are named and returned by `element_names(i)`
4. Add tests (ideally one of):

   * parse/validate tests
   * run/tap tests with a simple source/sink pipeline

### Adding runtime diagnostics

* Prefer adding fields to `DiagCtx` and `PipelineReport`
* If updates happen from streaming threads, use **atomics** (or another lock-free mechanism)
* Convert to plain snapshot types for reporting

---

## Dependency rules (non-negotiable)

* `builder/` should not depend on GStreamer or `pipeline/`
* `gst/` should not depend on `pipeline/`
* `nodes/` should not depend on `pipeline/` (Nodes are build-time descriptions, not runtime orchestrators)
* `pipeline/` is the orchestrator and can depend on `gst/`, `builder/`, `nodes/`, `contracts/`, `policy/`, `mpk/`

This keeps the architecture modular and prevents circular dependencies.

---

## Tests & examples

* `examples/` show typical end-to-end usage patterns:

  * decode RTSP
  * run MPK
  * run RTSP server
* `tests/` verify critical behaviors:

  * debug point behavior
  * encoding/decoding
  * file read paths
  * group expansion equivalence (input groups)
  * tensor output path + save/load round-trip
  * `modelsession_resnet50_guard_test` validates ModelSession accuracy and guard behavior

When adding features, prefer adding tests that:

* reproduce the pipeline string deterministically
* validate caps negotiation assumptions
* ensure failures produce useful `PipelineReport` diagnostics

---

## Design principles

1. **Determinism wins**

   * stable element names, stable pipeline strings, stable reports

2. **Debuggability is first-class**

   * bus logs, DOT dumps, boundary probes, clear reproduction steps

3. **Safe concurrency**

   * streaming-thread probes only touch atomics (snapshots produce plain reports)

4. **Never hang the process**

   * teardown is defensive; avoid blocking forever on broken plugin stacks

5. **Keep the public API stable**

   * internal refactors should not break user code unless intentionally versioned

```
