# NeatTensor Contract (RFC)

Goal: Make NeatTensor the only in/out type across build/run/stages/appsinks while
supporting dense N-D tensors, image/video semantics, accelerator layouts, and
zero-copy views without silent copies.

## Invariants (locked)

1) Stride units
   - strides are always expressed in bytes internally.

2) Shape ordering
   - Dense tensors: row-major default, last dimension contiguous when
     contiguous.
   - If ImageSpec is present: default layout is HWC unless explicitly set to
     CHW/HW/Planar.

3) Plane representation (explicit roles)
   - NV12: planes [Y, UV]
   - I420: planes [Y, U, V]
   - Each plane stores its role, shape, strides_bytes, and byte_offset.

4) Memory / mapping contract
   - map() returns an RAII object that keeps backing storage alive until unmap.
   - map() must be O(1) for CPU-owned/external buffers.
   - Non-mappable storage: map() fails or requires explicit CPU staging
     (cpu()/to_cpu_if_needed()).
   - Raw CPU pointer access requires an explicit mapping; `data()` is not
     exposed unless mapped.

5) Composite offset invariant (MUST)
   - Composite parent tensors must have byte_offset == 0.
   - Plane byte offsets are relative to the storage base.

6) Layout/shape consistency for image tensors
   - Interleaved RGB/BGR/GRAY8: layout=HWC, shape=[H,W,C] (C=1 or 3).
   - NV12/I420 composite: top-level shape=[H,W], layout=HW.
     Plane shapes are authoritative.

7) Mutability / aliasing
   - Default: tensors are immutable views.
   - Writable access requires `make_writable()` (copy-on-write if needed).
   - Aliasing rules must be explicit and consistent.

8) Device + stream semantics
   - DeviceType + device_id are always known.
   - Transfers are explicit and define sync behavior.

9) No silent copies
   - Any copy/convert/transfer must be explicit, or policy-driven with trace
     + metrics.
   - clone() is always deep copy.
   - cpu()/to_cpu_if_needed() copy only if required (traceable).
   - data_ptr<T>() only valid for CPU + dense + contiguous tensors.

## Conversion categories (vocabulary)

- reinterpret: metadata-only, safe only if provably identical bytes
- view: reshape/slice when strides allow
- pack: make contiguous (copy)
- convert: dtype or pixel format conversion (compute + new buffer)
- transfer: device copy + sync

## Conversion policies

- STRICT (default): fail if conversion is required.
- ALLOW_WITH_TRACE: allow with explicit trace + metrics.
- ALLOW_SILENT: prototyping only (avoid for production).

## MVP scope (v1)

- Dense tensors (U8/I8/BF16/FP32).
- ImageSpec: RGB, BGR, GRAY8, NV12, I420.
- Planes with explicit roles.
- CPU-backed storage + zero-copy cv::Mat views.
- GStreamer-backed storage (GstSample/GstBuffer) with deferred mapping.

Deferred
 - Full device transfer API + async streams.
 - DLPack export for composite planes.
 - Automatic conversion insertion beyond STRICT/ALLOW_WITH_TRACE.

## Definition of Done

- All stages accept/return NeatTensor.
- build()/run()/appsink return NeatTensor (or NeatTensorRef) only.
- No internal cv::Mat usage in pipeline graph.
- All conversions are explicit or policy-driven with trace/metrics.
