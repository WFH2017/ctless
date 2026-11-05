# Step 1 — NMRECON interface audit

Status: source inspection complete; numerical acceptance NOT RUN; integration NOT COMPLETE.

Scope: CT-less SPECT, with neural attenuation input and NMRECON physics. No training, existing engine modification, or active-job modification in this step.

## Verified source findings

- Existing adapter: `../ctless/nmrecon_system_matrix.py`. Its custom autograd forward calls NMRECON forward projection; backward calls NMRECON backprojection and returns a gradient for activity only. Plain `forward()` does not establish that custom autograd path.
- Engine source: `/home/wengfenghua/nmrecon_raw/nmrecon0708/NMRecon/nmrecon`.
- CLI exports forward projection, backprojection, and OSEM measurements. Current raw projection implementations load attenuation from configured CT/HU files (int16 conversion), or use zero attenuation. No direct floating-point neural mu input or mu VJP interface was found in these entry points.
- Attenuation header uses water coefficient 0.0154 with millimetre geometry, versus the learning convention 0.154 cm^-1. The integration boundary must explicitly convert mu_cm to mu_mm = mu_cm / 10, and apply the matching chain rule. Verify geometry/unit consistency numerically before accepting it.
- Existing adapter subset construction uses floor division and can omit remainder views. Replacement must partition all indices exactly once, including non-divisible view counts.
- Existing adapter infers projection dimensions from object dimensions. New interface must use actual detector metadata, and verify tensor layout with an asymmetric phantom.
- Existing source and binary fingerprints are recorded separately. Hashing both does NOT establish that the binary was built from this source.
- Current CT-less unroll imports the separate PyTomography projector implementation in `new1022.model`; the old NMRECON wrapper is not automatically used by it.

## Fixed staged acceptance plan

1. Interface audit (this document). No numerical correctness claim.
2. Isolated engine build + adapter: accept float32 activity and predicted mu directly, explicit units/layout, no CT fallback; expose NMRECON forward and true transpose; cover all views. Compare fixed-mu outputs with the original engine on GE and Siemens validation examples. Use a separate executable, not the running production binary.
3. Gradient acceptance: implement and check activity VJP and mu VJP against directional finite differences of the actual discretized forward operator. Test adjoint identity with signed probes, full views and subsets, CDR on/off. Report raw errors and an epsilon sweep; establish tolerances before examining clinical performance. Do not substitute straight-through or PyTomography gradients and call them NMRECON derivatives.
4. PyTomography integration: use its SystemMatrix/likelihood orchestration with NMRECON operators. Compare matched initialization, measurement/scatter scaling, subset order and iteration count with native NMRECON. First-order forward gradients alone do not imply differentiable OSEM: test every forward/backprojection operation used by the intended unroll. Higher-order support is not assumed.
5. Minimal unroll: validation-only single-case forward/backward and small overfit check, with explicit gradient paths and fixed checkpoint/source manifests. No test-set tuning.
6. Full training only after preceding gates have recorded results and user review.

Runtime cost of a subprocess/file-based adapter must be measured before choosing a production training interface. It is suitable as a correctness reference, not assumed to be fast enough for full training.

No gate may be marked passed based solely on comments, historical notebook output, or the existence of an API. User requested review between stages; stop after each staged deliverable.
