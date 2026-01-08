What do you think about this that the other chat gave me? How could I provide this context to Codex?

## (A) Architecture explanation (deep + structured)

### 1) What this framework is

This is a **C++17 library for building, validating, running, and debugging GStreamer pipelines** with a stable, composable API.

It has three “faces”:

1. **Builder / Graph API**

   * Users assemble a pipeline as a sequence/graph of typed `Node`s (e.g., `RTSPInput`, `H264DecodeSima`, `CapsNV12SysMem`, `OutputAppSink`).
   * Nodes generate deterministic GStreamer fragments (and deterministic element names).

2. **Runtime API (PipelineSession)**

   * Turns a Node list into a `gst_parse_launch()` pipeline, manages lifecycle, enforces contracts, and exposes convenient frame/tap streams (`FrameStream`, `TapStream`).

3. **Debug/Validation/RTSP utilities**

   * Validation without playing, debug taps at named points, boundary probes, bus capture, DOT dumps, and an RTSP server mode.

**Who uses it**

* Internal and external developers who want to:

  * Quickly run a pipeline and pull frames in C++ (without writing raw GStreamer boilerplate).
  * Validate pipeline integrity and contracts in CI.
  * Debug “where it stalls” in real deployments.
  * Serve a media pipeline over RTSP (e.g., image → NV12 → encoder → payloader).

**Primary use-cases**

* **Decode / ingest**: File or RTSP → demux/parse → decode → convert/caps → appsink → C++ consumer.
* **Tap/debug**: Add `DebugPoint("X")` and `run_tap("X")` to pull raw bytes or tight-packed frames at that boundary.
* **Validate**: `validate()` builds and prerolls (PAUSED) to catch negotiation issues early.
* **Serve RTSP**: `run_rtsp()` runs a GLib loop in a thread and uses `appsrc` to push generated NV12 frames into the pipeline exposed at a mount.

---

### 2) Runtime model

#### Initialization

* `gst_init_once()` is the single entry-point for GStreamer global init (thread-safe `std::call_once`).
* Introspection helpers (`element_exists` / `require_element`) ensure required plugins are installed before launching.

**Policy**: *every public runtime entry* calls `gst_init_once()` (e.g., `run()`, `run_tap()`, `validate()`, `run_rtsp()`).

#### Pipeline lifecycle (client-run pipelines)

1. User builds `PipelineSession` by calling `add(...)` / `nodes::...`.

2. `PipelineSession::run()`:

   * Enforces “sink last” contract (must end in `OutputAppSink` for `run()`).
   * Builds pipeline string (optionally inserting boundary `identity` elements between nodes).
   * `gst_parse_launch(...)` creates a bin/pipeline.
   * Optionally enforces element naming contract.
   * Attaches optional boundary probes (opt-in via env).
   * Locates appsink by deterministic name (`mysink`).
   * Sets state to `PLAYING` and returns `FrameStream`.

3. `FrameStream::next(...)`:

   * Pulls samples from appsink using a sliced polling loop.
   * Drains bus and throws if any error is observed.
   * Maps buffers into a **zero-copy NV12 view** (`FrameNV12Ref`) or produces a copy (`FrameNV12`).

4. Teardown:

   * `FrameStream::close()` unrefs the appsink and stops/unrefs the pipeline using `stop_and_unref()`.
   * `stop_and_unref()` is intentionally defensive: state transitions can deadlock in buggy plugins; it performs teardown async and leaks on timeout to avoid hanging the process/CI.

#### Threading model

* **GStreamer internal threads**: the pipeline uses internal streaming threads for decoding, scheduling, etc.
* **User thread**: the caller thread does:

  * appsink polling (`gst_app_sink_try_pull_sample`)
  * periodic bus draining
* **Diagnostics thread-safety**:

  * Bus messages are collected into `DiagCtx::bus` under a mutex.
  * Boundary counters are incremented from pad probes (GStreamer streaming threads); they write to `BoundaryFlowStats` fields. Those stats must be treated as atomic-ish/benign races (or upgraded to atomics if you want strict correctness).

#### Ownership model (must be consistent across the library)

* **GStreamer objects**: reference-counted. Rule: *if you store a pointer beyond the scope where you obtained it, you must `gst_object_ref` it; you must unref on shutdown.*
* `FrameStream` and `TapStream` own:

  * `pipeline_` (as a `GstElement*` with a ref held by creation; destroyed via `stop_and_unref`)
  * `appsink_` (explicit `gst_object_unref` in `close`)
* `FrameNV12Ref` owns:

  * a `shared_ptr<SampleHolder>` which owns:

    * `GstSample*` (unref)
    * mapped `GstVideoFrame` (unmap)
* RTSP server uses heap-allocated callback contexts (`PushCtx`) deleted via the `unprepared` signal path, and holds `appsrc` via `gst_object_ref`.

---

### 3) Dataflow: frames in/out, conversions, timestamps, caps negotiation

#### Pipeline string generation

* Each `Node` produces:

  * `gst_fragment(node_index)` → fragment appended into a `gst-launch`-compatible string.
  * `element_names(node_index)` → deterministic element names used for naming enforcement and debug tooling.

`build_pipeline_full(...)`:

* Concatenates fragments with `!`
* Optionally inserts boundary markers:

  * `identity name=sima_b<i> silent=true`
  * Records `BoundaryFlowStats` for each boundary.

`build_pipeline_tap(...)`:

* Locates a `DebugPoint` by `kind()=="DebugPoint"` and `user_label()==<name>`.
* Builds a truncated pipeline up to that node.
* Appends:

  * `appsink name=tap_<dbg> emit-signals=false sync=false max-buffers=1 drop=true`

#### Output via appsink (`FrameStream`)

* `run()` expects **NV12 + SystemMemory** (explicit contract).
* In `FrameStream::next()`:

  * Pull sample.
  * Validate caps:

    * `gst_video_info_from_caps`
    * `GST_VIDEO_FORMAT_NV12`
    * `require_system_memory_or_throw(...)` ensures not NVMM/device memory.
  * Map buffer into `GstVideoFrame` (read).
  * Return `FrameNV12Ref` containing:

    * `y`, `uv` pointers + strides
    * timestamps: `PTS/DTS/DURATION` copied from `GstBuffer`
    * `keyframe` detection via `GST_BUFFER_FLAG_DELTA_UNIT`

`next_copy()` produces a contiguous NV12 buffer (tight pack) by copying row-by-row, respecting strides.

#### Tap output (`TapStream`)

* Tap is intentionally more permissive:

  * It reports caps string + memory features.
  * It attempts a **tight pack** for a subset of raw formats using `gst_video_frame_map`:

    * NV12, I420, RGB, BGR, GRAY8
  * If it can’t pack, it tries `gst_buffer_map` and copies raw bytes.
  * If mapping fails (typical on DMA/NVMM), it returns:

    * `memory_mappable=false` and a human-readable reason.

#### Timestamp semantics

* Client-run pipelines:

  * timestamps are whatever upstream provides.
  * In debug/analysis, stalls are diagnosed with wall-clock monotonic time (`g_get_monotonic_time`) alongside buffer PTS.

* RTSP server:

  * pushes synthetic frames with:

    * `PTS = frame_count * frame_duration_ns`
    * `DTS = PTS`
    * `DURATION = frame_duration_ns`
  * `appsrc` is configured:

    * `is-live=true`
    * `format=GST_FORMAT_TIME`
    * `do-timestamp=false` (we provide timestamps ourselves)

#### Caps negotiation strategy

* The library makes negotiation explicit through Nodes:

  * e.g., `CapsNV12SysMem(w,h,fps)` inserts a capsfilter requiring `video/x-raw(memory:SystemMemory),format=NV12,...`
* Any pipeline that intends to be consumed by `FrameStream` should end with such a caps node (or a decoder node that already enforces system memory output).

---

### 4) RTSP server integration

RTSP server is an **alternate runtime path** driven by GLib main loop + GstRTSPServer.

Key behaviors in your implementation:

* `PipelineSession::run_rtsp(opt)`:

  * Requires `appsrc`, `rtph264pay`, `h264parse`.
  * Finds the `AppSrcImage` node to obtain:

    * encoded dimensions and fps
    * precomputed NV12 frame bytes shared via `shared_ptr<vector<uint8_t>>`
  * Builds a launch string wrapped as: `"( <node0> ! <node1> ! ... )"`

* Server thread:

  * Creates `GstRTSPServer`, mounts a `GstRTSPMediaFactory`.
  * `gst_rtsp_media_factory_set_shared(factory, FALSE)`

    * each client gets its own media/pipeline instance.
  * On `"media-configure"`:

    * obtains the top element and finds `appsrc` named `"mysrc"`.
    * sets caps and appsrc properties.
    * allocates a per-media `PushCtx`:

      * holds a ref to appsrc
      * holds NV12 shared buffer
      * schedules a periodic `g_timeout_add` to push buffers.
    * hooks `"unprepared"` to cleanup, cancel timer, unref appsrc, delete ctx.

**Client connection path**

* Clients connect to `rtsp://127.0.0.1:<port>/<mount>`.
* When a client requests media, RTSP server instantiates a new pipeline from the factory launch string and begins calling your timer-driven push callback.

**Where pipelines attach**

* The RTSP pipeline is the factory launch string itself.
* Your code binds into that pipeline at `"mysrc"` inside `"media-configure"`.

---

### 5) Error handling & logging strategy

#### Bus handling

The design is: **bus polling is cheap and always-on** in run/tap.

* `drain_bus(pipeline, diag)`:

  * pops all available bus messages
  * stores them into `DiagCtx::bus` (type/src/detail + wall_time_us)
* `throw_if_bus_error(pipeline, diag, where)`:

  * scans bus messages and throws immediately on `GST_MESSAGE_ERROR`
  * on error:

    * optionally dumps DOT (tagged by `where`)
    * throws `PipelineError` carrying a `PipelineReport`

This gives you:

* structured error surfaces (exception contains a report)
* stable reproduction hints (gst-launch string + env suggestions)
* an always-on trail of relevant bus events

#### Exceptions vs status

* **Runtime “run/tap”**: throw exceptions on fatal errors (parse failure, missing sink, bus error).
* **validate()**: returns a `PipelineReport` describing failures instead of throwing to keep CI-friendly.
* **run_debug()**: catches both `PipelineError` and generic exceptions and returns a report.

This is a deliberate tradeoff: developers can “fail fast” in normal flows, but still have a “report mode” for tests and diagnostics.

---

### 6) Debugging hooks & observability

Your framework already has the right “knobs”—the refactor should preserve them exactly:

* DOT dumps:

  * enabled via `SIMA_GST_DOT_DIR`
  * used on state failures, errors, and heavy snapshots
* Boundary probes:

  * activated by `SIMA_GST_BOUNDARY_PROBES=1`
  * boundary insertion is mode-dependent:

    * `SIMA_GST_RUN_INSERT_BOUNDARIES`
    * `SIMA_GST_TAP_INSERT_BOUNDARIES`
    * `SIMA_GST_VALIDATE_INSERT_BOUNDARIES`
  * `boundary_summary()` identifies “likely stall” boundaries by last activity time.
* Naming contract enforcement:

  * `SIMA_GST_ENFORCE_NAMES=1`
  * ensures every element in the parsed bin is accounted for by some node’s `element_names()`
* Sliced appsink polling:

  * `SIMA_GST_POLL_SLICE_MS` for responsiveness and ongoing bus/error checks
  * `SIMA_GST_TIMEOUT_RETURNS_NULL` to choose between nullopt vs exception on timeout
* Heavy report snapshots:

  * DOT dump + a caps dump of bin element names (and optionally extended later)

---

### 7) Key design principles & tradeoffs

**Principles**

1. **Stable public API, volatile internals**

   * `include/pipeline/PipelineSession.h` is stable.
   * Internals can move freely as long as behavior and signatures remain unchanged.

2. **Deterministic names**

   * Element naming contract is essential for:

     * “get element by name”
     * debug tap insertion
     * boundary probes
     * reliable diagnostics

3. **Explicit memory contract where needed**

   * `FrameStream` is strict (NV12 + SystemMemory) because zero-copy CPU access must be safe.
   * `TapStream` is flexible because debug is exploratory.

4. **Never hang the process**

   * `stop_and_unref()` chooses “leak pipeline” over hanging CI due to known plugin deadlocks.

5. **Observability is first-class**

   * You always collect bus messages.
   * You can optionally attach probes and generate DOT graphs.

**Tradeoffs**

* Using `gst_parse_launch` is simple and flexible, but makes “strong typing” harder. The node interface is the bridge: typed API → launch string.
* Boundary stats are lightweight but not perfectly synchronized; “good enough to locate stall points” beats heavy locking in hot paths.
* Async teardown is pragmatic; it’s not pretty, but it prevents deadlocks from taking down end-to-end tests.

---

### 8) What goes wrong (common failure modes)

1. **Caps mismatch at appsink**

   * Symptoms: `FrameStream::next` throws “expected NV12”.
   * Causes:

     * missing/incorrect capsfilter before the sink
     * decoder outputs I420/RGB
   * Fix:

     * add `CapsNV12SysMem(w,h,fps)` or ensure decoder node enforces NV12.

2. **Non-SystemMemory buffers**

   * Symptoms: `require_system_memory_or_throw` triggers.
   * Causes:

     * upstream produces NVMM / DMA buffers (Jetson-style)
   * Fix:

     * insert conversion/download elements or enforce `(memory:SystemMemory)` in caps before the DebugPoint / appsink.
     * ensure decoder node converts to system memory (your sima decoder already does via capsfilter).

3. **Negotiation stalls / preroll timeouts**

   * Symptoms:

     * validate() times out pulling preroll
     * run() returns nullopt repeatedly (if configured)
   * Causes:

     * live source without data
     * missing depay/parse step
     * wrong caps at a join
   * Fix:

     * enable boundary probes + boundary insertion; check `boundary_summary`.
     * check bus messages in report snapshot.

4. **State-change deadlocks in plugins**

   * Symptoms:

     * teardown hangs inside `gst_element_set_state(NULL)`
   * Mitigation:

     * `stop_and_unref()` async teardown with timeout.
   * Follow-up:

     * you can optionally add a “kill switch” policy later to aggressively abort.

5. **RTSP server main loop issues**

   * Symptoms:

     * server reports “running” but no media
     * callbacks not firing
   * Causes:

     * not attaching server, or main loop not started
     * push timer stopped early due to flow return
   * Fix:

     * check `"media-configure"` path and `gst_app_src_push_buffer` return values.
     * confirm mount path formatting (`/image` default).

6. **Buffer lifetime bugs**

   * Symptoms:

     * crashes or corrupted frames when user holds references
   * Root cause:

     * returning raw pointers without owning the underlying sample mapping
   * Your mitigation:

     * `FrameNV12Ref` holds `shared_ptr<SampleHolder>` that owns mapping + sample.

7. **Latency / backpressure**

   * Symptoms:

     * high latency or stuttering
   * Causes:

     * missing queues, too-large buffers, sync=true, drop=false
   * Your choices:

     * appsink configured `max-buffers=1 drop=true sync=false` for low latency.

---

### 9) Dependency diagram (text)

Modules in your repository (based on your tree) and allowed dependencies:

```
[pipeline] PipelineSession / FrameStream / TapStream / Errors / PipelineReport
   |--> [builder] Graph / Builder / Node / GraphPrinter
   |--> [nodes] typed node implementations + node groups
   |--> [gst] GstInit / GstHelpers / GstParseLaunch / GstBusWatch / GstPadTap / GstIntrospection
   |--> [policy] DefaultPolicy, MemoryPolicy, EncoderPolicy, DecoderPolicy, RtspPolicy
   |--> [contracts] Validators / ContractRegistry / ValidationReport
   \--> [mpk] MpKLoader / MpKPipelineAdapter (optional integration)

[nodes] depends on:
   |--> [builder] Node interface
   |--> [gst] (only for introspection helpers, optional)
   \--> OpenCV (ONLY for AppSrcImage implementation)

[gst] depends on:
   \--> GStreamer / GLib only (NO pipeline/builder/nodes)

[builder] depends on:
   \--> STL only (NO GStreamer)

[contracts] depends on:
   \--> builder + policy (+ STL)

[policy] depends on:
   \--> STL only (or minimal)
```

Hard rule: **`gst/` must not depend on `pipeline/`** (no circular diagnostics exceptions leaking downward). `pipeline/` is the top-level orchestrator.

---

## (B) Concrete refactor plan

You already have the correct high-level folder structure and headers in `include/`. The refactor plan below is **prescriptive**: it matches your current tree and explains exactly what each file “owns,” so the monolith becomes a set of clean, testable modules.

### 1) Proposed folder structure (matches your repo)

```
include/
  gst/            # public gst utilities (thin)
  pipeline/       # public runtime API: PipelineSession, FrameStream, TapStream, Reports, Errors
  nodes/          # public node APIs
  builder/        # public builder graph API
  policy/         # public policy knobs
  contracts/      # public validation layer
  mpk/            # model-pack adapter layer

src/
  gst/            # implementations of include/gst
  pipeline/       # implementations of include/pipeline
  nodes/          # implementations of include/nodes
  builder/        # implementations of include/builder
  policy/         # implementations
  contracts/      # implementations
  mpk/            # implementations

old_PipelineSession.cpp  # kept only during migration; deleted at end
```

### 2) New file list (headers + cpp) with responsibilities

You already have most of these in place; this is the “ownership map” Codex must follow.

#### `gst/` module

* `include/gst/GstInit.h` + `src/gst/GstInit.cpp`
  Owns: `gst_init_once()`

* `include/gst/GstHelpers.h` + `src/gst/GstHelpers.cpp`
  Owns:

  * env helpers: `env_bool`, `env_str`
  * time helper: `now_mono_us`
  * string helpers: `sanitize_name`, `json_escape`
  * caps formatting: `gst_caps_to_string_safe`, `gst_structure_to_string_safe`, `caps_features_string`
  * DOT dump: `maybe_dump_dot`

* `include/gst/GstIntrospection.h` + `src/gst/GstIntrospection.cpp`
  Owns: `element_exists`, `require_element`

* `include/gst/GstBusWatch.h` + `src/gst/GstBusWatch.cpp`
  Owns:

  * `gst_message_to_string`
  * `drain_bus`
  * `throw_if_bus_error`
    (and **no** dependency on pipeline errors: return status or accept callbacks; pipeline layer converts into exceptions)

* `include/gst/GstPadTap.h` + `src/gst/GstPadTap.cpp`
  Owns:

  * boundary probe ctx + attach
  * boundary summary builder

* `include/gst/GstParseLaunch.h` + `src/gst/GstParseLaunch.cpp`
  Owns:

  * thin wrapper around `gst_parse_launch` to normalize errors and return a `GstElement*`

#### `pipeline/` module

* `include/pipeline/PipelineReport.h` + `src/pipeline/PipelineReport.cpp`
  Owns: `PipelineReport::to_json()`

* `include/pipeline/Errors.h` + `src/pipeline/Errors.cpp`
  Owns: `PipelineError` (exception + embedded report)

* `include/pipeline/TapStream.h` + `src/pipeline/TapStream.cpp`
  Owns:

  * `TapStream` runtime object
  * `TapPacket` packing logic (calls gst helpers)
  * tight-pack helpers (`infer_tap_format_and_meta`, `pack_raw_video_tight`)

* `include/pipeline/PipelineSession.h` + `src/pipeline/PipelineSession.cpp`
  Owns:

  * session orchestration: `run`, `run_tap`, `validate`, `run_rtsp`, `run_debug`
  * build pipeline strings using nodes (or calls into a `builder` adapter)
  * enforce sink-last
  * naming contract enforcement (or moved to `contracts/` if you prefer)
  * keeps `last_pipeline_`

* `src/pipeline/internal/DiagCtx.h` (private header; **not** installed)
  Owns:

  * `DiagCtx` definition and `snapshot_basic()`
  * `BoundaryFlowStats` storage strategy
  * makes diagnostics sharable without polluting public headers

* `src/pipeline/internal/StopAndUnref.h/.cpp` (private)
  Owns:

  * `stop_and_unref()` teardown watchdog

* `src/pipeline/internal/RtspServerImpl.*` (private)
  Owns:

  * `RtspServerImpl`, `PushCtx`, callbacks
  * `RtspServerHandle` behavior (public type still in include/pipeline or include/policy if you keep it there)

#### `nodes/` module

* Each header under `include/nodes/**` must have a matching `.cpp` under `src/nodes/**`:

  * `nodes/common/*` → simple fragments and deterministic element names
  * `nodes/io/AppSrcImage.cpp` → OpenCV + NV12 packing/padding (only file that needs OpenCV)
  * `nodes/rtp/*` → depay/pay
  * `nodes/sima/*` → sima encoder/decoder/parse elements
  * `nodes/groups/*` → composition helpers that return vectors of nodes or a `NodeGroup`

**Rule**: node `.cpp` files must not include pipeline headers.

---

### 3) Public vs private headers policy

* **Public** (`include/**`):

  * stable API types: `PipelineSession`, `FrameStream`, `TapStream`, `PipelineReport`, node types, builder graph, contracts, policy.
  * must avoid heavy includes: forward declare GStreamer structs whenever possible.

* **Private** (`src/**/internal/**`):

  * any “glue structs” like `DiagCtx`, `BuildResult`, `SampleHolder`, RTSP callback contexts.
  * anything that would otherwise create circular dependencies or balloon compile times.

**Non-negotiable**: private headers must never be included from public headers.

---

### 4) Forward declarations vs includes rules

**Public headers**

* Forward declare:

  * `struct _GstElement; struct _GstCaps; struct _GstSample; struct _GstBuffer;`
* Do **not** include `<gst/gst.h>` or `<opencv2/opencv.hpp>` in public headers.

**.cpp files**

* Include your own header first (`#include "..."`).
* Then include only what you use.
* Keep OpenCV restricted to `AppSrcImage.cpp` (and any optional image utilities).

---

### 5) Naming conventions, namespaces, ownership rules

**Namespaces**

* Public API lives in `namespace sima`.
* Node factories live in `namespace sima::nodes`.
* Private helper namespaces are allowed but should be inside `sima` (e.g., `sima::detail`).

**Element names**

* Must remain deterministic:

  * `mysink`, `mysrc`, `pay0`, and `n<idx>_<role>` patterns.
* Any refactor must not change these names unless you intentionally version the naming contract.

**Ownership rules**

* If a struct stores a `GstElement*` longer than the local scope, it must hold a ref.
* Callback user_data must be heap-allocated and freed exactly once via a destroy notify or a “unprepared” handler.
* `FrameNV12Ref` must always keep the underlying mapping alive via a shared holder.

---

### 6) Order of operations to refactor safely (step-by-step)

This is the safest migration plan from `old_PipelineSession.cpp` into your existing `src/` layout with **no behavior changes**:

1. **Move pure helpers first**

   * Extract `gst_init_once`, env helpers, JSON escape, sanitize name, caps stringify, DOT dump into `src/gst/*`.
   * Zero behavior changes; just move code.

2. **Extract bus watch**

   * Move `gst_message_to_string`, `drain_bus`, `throw_if_bus_error` into `src/gst/GstBusWatch.cpp`.
   * Ensure pipeline layer still throws `PipelineError` with the same text/report.

3. **Extract boundary probes**

   * Move `BoundaryProbeCtx`, `boundary_probe_cb`, `attach_boundary_probes`, `boundary_summary` into `src/gst/GstPadTap.cpp`.
   * Keep env keys and defaults unchanged.

4. **Extract teardown watchdog**

   * Move `stop_and_unref` into `src/pipeline/internal/StopAndUnref.*`.
   * Keep timeout, behavior (leak on timeout), and warning message unchanged.

5. **Extract diagnostics context**

   * Move `DiagCtx` + `snapshot_basic()` into `src/pipeline/internal/DiagCtx.h`.
   * Keep report population identical.

6. **Extract TapStream**

   * Move `TapStream` + packing helpers into `src/pipeline/TapStream.cpp`.
   * Ensure `TapPacket` fields identical and `memory_mappable` behavior unchanged.

7. **Extract FrameStream**

   * Move `FrameStream` implementation into `src/pipeline/PipelineSession.cpp` or a dedicated `FrameStream.cpp` if you have that header.
   * Keep sliced polling behavior and env flags unchanged.

8. **Extract nodes**

   * Move each node implementation into matching `src/nodes/**` file, one file per node category.
   * Preserve `gst_fragment()` output exactly (including spacing and element names).

9. **Extract RTSP server**

   * Move `RtspServerImpl`, `PushCtx`, callbacks into `src/pipeline/internal/RtspServerImpl.*`.
   * Keep mount default, URL format, caps setting, timer behavior unchanged.

10. **Slim PipelineSession.cpp**

* After extraction, `PipelineSession.cpp` should contain only:

  * orchestration logic
  * pipeline build calls
  * name enforcement call
* No OpenCV, no big helper structs.

11. **Delete old_PipelineSession.cpp**

* Only after tests pass and the new pipeline module is functionally identical.

---

### 7) Definition of done checklist

* ✅ `old_PipelineSession.cpp` removed from build and deleted (or kept only as reference outside compilation).
* ✅ All tests in `tests/` pass.
* ✅ `PipelineSession.h` public API unchanged (unless explicitly requested).
* ✅ Element names and pipeline strings match baseline outputs (where applicable).
* ✅ Env flags preserved:

  * `SIMA_GST_DOT_DIR`, `SIMA_GST_BOUNDARY_PROBES`, `SIMA_GST_*_INSERT_BOUNDARIES`,
  * `SIMA_GST_ENFORCE_NAMES`, `SIMA_GST_POLL_SLICE_MS`, `SIMA_GST_TIMEOUT_RETURNS_NULL`, etc.
* ✅ No public header includes OpenCV or `gst/*.h`.
* ✅ No `gst/` code depends on `pipeline/` headers (no circular deps).
* ✅ `AppSrcImage.cpp` is the only compilation unit that includes OpenCV headers.
* ✅ `clang-tidy`/warnings not worse than before (if enabled).
* ✅ Leak-on-timeout teardown behavior preserved exactly.
