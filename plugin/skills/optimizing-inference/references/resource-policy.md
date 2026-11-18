# Compute resource policy

Set `goals.resource_policy` when the user specifies how the inference workload may use the device. Ask which outcome matters when throughput, latency, energy and interactive responsiveness compete. Keep the registered workload, quality limits, resource authorization and target hardware in either mode.

## Maximum performance

Set `maximum_performance` to maximize the stated metric under the stated limits. Inventory authorized CPU cores and accelerators. Verify which runtimes can place the pinned model or its stages on each unit. Benchmark the strongest tuned single-unit placement in addition to promising combinations.

Try parallel requests on separate units, overlapping independent stages across requests, and supported in-process batching or pipelining. Measure useful stage overlap and wall time including transfers, synchronization, queueing and output. Reuse read-only weights or share tensor buffers where the runtime supports it and verify buffer placement, copies and lifetime. Measure sustained bandwidth, thermals and memory pressure where possible. Select the fastest configuration that meets the registered limits. Record actual placement, fallback and any units that could not run the model.

## Shared-device efficiency

Set `shared_device_efficiency` to meet the stated workload targets while leaving capacity for other applications. Agree on what should remain responsive, such as a game, a call, media playback or a desktop application. Register a representative foreground workload, its performance with inference idle and any latency or energy limits the user specifies.

Screen fewer inference workers and CPU threads, bounded queues, capped concurrency, OS scheduling hints, or a supported accelerator that leaves resources available to the foreground work. Compare on the same device with the foreground workload present in each measured run. Record inference latency/throughput and foreground latency or missed deadlines, plus CPU usage, memory, thermal state and energy when those can be measured. Prefer the configuration that consumes fewer of the constrained resources while meeting both workloads' targets. If foreground contention was never measured, report device responsiveness as UNPROVEN.

## Platform implications

The objectives apply to phones, laptops and desktops. Actual placement depends on the model, runtime, memory topology and device access. A shared memory pool can still have competing bandwidth, copies or synchronization. Discrete GPUs introduce a separate device memory budget and transfers. Test the real path on the deployed hardware.

- Qualcomm Android can expose NPU or GPU backends through an appropriate QNN build. Verify graph placement and fallback. [ONNX Runtime QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- Android scheduling and thermal hints can help bound concurrent application impact. [Android Dynamic Performance Framework](https://developer.android.com/games/optimize/adpf)
- Apple Core ML allows different compute-unit policies and provides a compute plan. [Apple Core ML performance analysis](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode)
- On discrete NVIDIA systems, measure transfers and supported copy/compute overlap. [CUDA memory optimization guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
