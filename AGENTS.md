
# AGENTS.md — PipelineSession refactor rules (Codex)

You are refactoring a monolithic `old_PipelineSession.cpp` into the existing multi-file repository structure under `src/` and `include/`.

## Non-negotiable constraints

1) **Do NOT change public API**:
   - Treat `include/pipeline/PipelineSession.h` (and all public headers under `include/`) as **stable** unless explicitly asked.
   - Do not rename public types, functions, fields, or namespaces.

2) **No behavior changes**:
   - Preserve runtime behavior, env var keys/defaults, pipeline string formatting, element names, error messages, and timeouts.
   - Refactor = move code + reduce coupling, not feature work.

3) **Incremental steps**:
   - Make small, reviewable commits.
   - After each step: build + run tests.
   - If a step breaks tests, revert that step or fix before continuing.

4) **No circular dependencies**:
   - `src/gst/*` must not include or depend on `pipeline/*`.
   - `nodes/*` must not include or depend on `pipeline/*`.

5) **Minimal includes**:
   - Public headers: forward declare GStreamer types; do not include `<gst/gst.h>` or OpenCV.
   - OpenCV includes must be confined to `src/nodes/io/AppSrcImage.cpp` (and any dedicated image utility .cpp if required).

6) **Ownership safety (must not regress)**:
   - Any stored `GstElement*` beyond a local scope must be ref-counted via `gst_object_ref` and unref’d.
   - Callback `user_data` must be freed exactly once using destroy-notify or `unprepared` cleanup.
   - `FrameNV12Ref` must retain mapped memory via an owning holder object.

## Build & test commands

If unknown, use these placeholders and adjust to match project conventions:

- Configure:
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build:
  - `cmake --build build -j`
- Test:
  - `ctest --test-dir build --output-on-failure`

If the repo uses a different build system or build dir name, detect it from existing CI or README and update the commands locally in commits (do not guess silently).

## Target module layout (must follow)

- `src/gst/`: GStreamer/GLib-only helpers
  - GstInit, GstHelpers, GstIntrospection, GstBusWatch, GstPadTap, GstParseLaunch
- `src/pipeline/`: public runtime implementations
  - PipelineSession.cpp, TapStream.cpp, PipelineReport.cpp, Errors.cpp
  - plus private internals under `src/pipeline/internal/`
- `src/nodes/`: Node implementations grouped by category
  - `nodes/common`, `nodes/io`, `nodes/rtp`, `nodes/sima`, `nodes/groups`

Public headers are already present under `include/` and must remain stable.

## Refactor plan (execute in order)

### Step 1 — Extract pure helpers into gst module
Move (no edits besides symbol visibility and header wiring):
- `gst_init_once` -> `src/gst/GstInit.cpp`
- env helpers, `sanitize_name`, `json_escape`, caps stringify, `maybe_dump_dot` -> `src/gst/GstHelpers.cpp`
- `element_exists` / `require_element` -> `src/gst/GstIntrospection.cpp`

Acceptance:
- Build succeeds
- All tests pass

### Step 2 — Extract bus utilities
Move:
- `gst_message_to_string`
- `drain_bus`
- `throw_if_bus_error`

into `src/gst/GstBusWatch.cpp`.

Acceptance:
- All runtime call sites behave identically
- Tests pass

### Step 3 — Extract boundary probe utilities
Move:
- `BoundaryProbeCtx`
- `boundary_probe_cb`
- `attach_boundary_probes`
- `boundary_summary`

into `src/gst/GstPadTap.cpp`.

Acceptance:
- Env flags preserved (names + defaults)
- Tests pass

### Step 4 — Extract teardown watchdog
Move `stop_and_unref` into `src/pipeline/internal/StopAndUnref.*`.

Acceptance:
- Timeout/leak behavior unchanged
- Tests pass

### Step 5 — Extract diagnostics context
Move `DiagCtx` into `src/pipeline/internal/DiagCtx.h` (private header).
Ensure PipelineReport snapshot fields remain identical:
- pipeline_string, nodes, bus, boundaries, repro strings

Acceptance:
- Tests pass

### Step 6 — Extract TapStream implementation
Move TapStream logic + raw packing helpers into `src/pipeline/TapStream.cpp`.
Preserve:
- caps parsing behavior
- memory mappability reporting
- timestamps + keyframe detection

Acceptance:
- Tests pass

### Step 7 — Extract FrameStream implementation
If FrameStream has its own header, place it in `src/pipeline/FrameStream.cpp`; otherwise keep inside `src/pipeline/PipelineSession.cpp` but isolated.
Preserve:
- sliced polling logic
- timeout semantics governed by env vars
- NV12 + SystemMemory contract enforcement

Acceptance:
- Tests pass

### Step 8 — Move node implementations into src/nodes
For each public node header under `include/nodes/**`, ensure there is a corresponding .cpp that owns the implementation.
Rules:
- Do not change `gst_fragment()` output or `element_names()`.
- Do not introduce pipeline dependencies.

Acceptance:
- Tests pass

### Step 9 — Extract RTSP server internals
Move RTSP server structs + callbacks into `src/pipeline/internal/RtspServerImpl.*`.
Preserve:
- mount path rules
- url formatting
- appsrc caps and timestamp behavior
- shared=false factory
- cleanup via `unprepared`

Acceptance:
- Tests pass

### Step 10 — Slim PipelineSession.cpp
PipelineSession.cpp should be orchestration only:
- build pipeline string
- parse launch
- enforce naming (if enabled)
- set state, create streams, call helpers

Acceptance:
- Tests pass

### Step 11 — Remove old_PipelineSession.cpp
Remove from build system and delete the file once parity is verified.

Acceptance:
- Tests pass
- No references remain

## Acceptance checks (must be satisfied before finishing)

- `PipelineSession.h` unchanged
- Element names preserved: `mysink`, `mysrc`, `pay0`, and all `n<idx>_*`
- Env vars preserved: `SIMA_GST_*` keys and defaults unchanged
- `include/` contains no OpenCV or gst header includes
- `src/gst/` depends only on GStreamer/GLib + STL
- All tests under `tests/` pass
- No new warnings that break existing warning policy

## If tests fail

1) Identify the failing test and isolate the refactor step that caused it.
2) Compare pipeline strings and element names against the pre-refactor behavior.
3) Check lifetime/refcounting changes around appsink/appsrc/sample mapping.
4) Revert the offending step if necessary and redo it with smaller diffs.
5) Do not proceed to the next step until tests are green.