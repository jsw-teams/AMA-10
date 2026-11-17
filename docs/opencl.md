# OpenCL inference backend

The optional OpenCL path parallelizes one output neuron per work item for the shared recurrent cell and one vocabulary logit per work item for the output projection. It keeps weights resident in device buffers for the lifetime of an inference session. The current public state remains host-owned, so each token copies fast/slow state to the device and copies state and logits back. The activity-based halt decision is evaluated on-device after a recurrent step; this preserves the CPU stopping rule.

With `--opencl-adamw`, training runs recurrent sequence forward propagation, the output projection, full reverse-time recurrent BPTT, and clipped AdamW updates on the GPU; the CPU handles data loading and loss orchestration. Token IDs and initial recurrent state transfer once per microbatch. The trajectory cache remains on-device between forward and backward, while the per-token final hidden vectors return for loss calculation. Recurrent reverse steps accumulate parameter gradients and carry fast/slow derivatives on-device. AdamW moments persist on device between updates and synchronize before checkpoint writes. The OpenCL test compares all eight accumulated parameter-gradient groups with the CPU path, including an identical short sequence, and separately compares optimizer parameters and moments. End-to-end trials show the hybrid still slower at widths 64–512, with a smaller gap at wider dimensions.

## Build and verify

On Windows with an OpenCL ICD and SDK headers/import library available under `CUDA_PATH`:

```powershell
.\scripts\build.ps1 -OpenCL -Test
```

This compiles the CLI, the standalone benchmark, and the OpenCL parity test. The parity test covers fixed maximum depth and early stopping and compares logits/state with a relative tolerance of `3e-4`; it also checks full recurrent training-gradient parity, output-head finite differences, AdamW updates, and fresh/resumed training.

## Initial measurement

On 2026-11-16, the host reported an NVIDIA GeForce RTX 4050 Laptop GPU with 6,141 MiB total memory. One warm-state 500-token run at width 256, input width 256, vocabulary 4096, and depth two measured:

| Backend | Time | Throughput |
|---|---:|---:|
| CPU | 0.267 s | 1,871 tokens/s |
| OpenCL GPU | 0.195 s | 2,566 tokens/s |

This single run is a smoke benchmark, not a stable performance estimate. The small shipped width-32/vocabulary-512 checkpoint takes about 77 ms through CPU generation versus 747 ms with OpenCL, including process startup and OpenCL program compilation; the GPU backend is slower for that small workload. Both paths produced byte-identical greedy output in that run. Kernel-launch and host/device transfer costs dominate small models; persistent device-side state and fused work should be measured before claiming general speedups.
