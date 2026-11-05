// Practical DRAM bandwidth of the GPU: the "100%" that decode kernels should be
// compared against (datasheet bandwidth is a theoretical peak no kernel reaches).
//
// Tests, each printed as one JSON object per line on stdout:
//   device      name, SM count, L2 size, bus width, memory clock, theoretical GB/s
//   read        read-only stream over N bytes (16-byte non-caching loads, 4 in flight
//               per thread); L2 is flushed before every timed run so small buffers
//               measure DRAM, not cache
//   copy        read N/2 bytes + write N/2 bytes (counts N bytes moved)
//   graph_read  a CUDA graph of 32 back-to-back launches, each reading its own S-byte
//               region: the best a kernel of that size can do. The gap to "read" is
//               the per-launch ramp-up and tail, which is why small decode kernels
//               rarely reach peak bandwidth. With PDL (programmatic dependent launch,
//               sm_90+) the next launch starts its prologue while the previous drains.
//
// Build: nvcc -O3 -std=c++17 -arch=native -o bw_peak bw_peak.cu
// Usage: bw_peak [--sizes-mib 8,32,128,512,1024] [--graph-mib 1,4,16] [--iters 30]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                                              \
  do {                                                                                        \
    cudaError_t err_ = (x);                                                                   \
    if (err_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_)); \
      std::exit(1);                                                                           \
    }                                                                                         \
  } while (0)

// 16-byte load that bypasses L1 (we never reuse the data, so caching it only costs).
__device__ __forceinline__ uint4 load_stream(const uint4* p) {
#if __CUDA_ARCH__ >= 700
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
#else
  return __ldg(p);
#endif
}

// Grid-stride read. UNROLL independent loads are issued before any is used, so each
// thread keeps several memory requests in flight (memory-level parallelism).
template <int UNROLL>
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned* sink, int pdl) {
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  unsigned acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = load_stream(p + i + u * stride);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    const uint4 v = load_stream(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
#if __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (acc == 0x9e3779b9u) sink[0] = acc;  // data-dependent store: the compiler cannot drop the loads
}

__global__ void copy_kernel(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) b[i] = a[i];
}

// Writes a buffer larger than L2 so the next timed kernel starts with a cold cache.
__global__ void flush_kernel(float* f, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) f[i] += 1.0f;
}

static float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static std::vector<size_t> parse_list(const char* s) {
  std::vector<size_t> out;
  std::string str(s);
  size_t pos = 0;
  while (pos < str.size()) {
    size_t comma = str.find(',', pos);
    if (comma == std::string::npos) comma = str.size();
    out.push_back(std::strtoull(str.substr(pos, comma - pos).c_str(), nullptr, 10));
    pos = comma + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  std::vector<size_t> sizes_mib = {8, 32, 128, 512, 1024};
  std::vector<size_t> graph_mib = {1, 4, 16};
  int iters = 30;
  for (int a = 1; a + 1 < argc; a += 2) {
    if (!std::strcmp(argv[a], "--sizes-mib")) sizes_mib = parse_list(argv[a + 1]);
    else if (!std::strcmp(argv[a], "--graph-mib")) graph_mib = parse_list(argv[a + 1]);
    else if (!std::strcmp(argv[a], "--iters")) iters = std::max(5, std::atoi(argv[a + 1]));
  }
  const int launches_per_graph = 32;

  int dev = 0, sms = 0, l2 = 0, bus_bits = 0, mem_khz = 0, cc_major = 0, cc_minor = 0;
  CHECK(cudaSetDevice(dev));
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
  CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
  // Memory clock and bus width are informational; some toolkits deprecate them, so ignore errors.
  if (cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev) != cudaSuccess) bus_bits = 0;
  if (cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess) mem_khz = 0;
  cudaGetLastError();
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  // DDR: two transfers per clock. bytes/s = 2 * clock(Hz) * bus width(bytes).
  const double theoretical_gbs = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  std::printf("{\"test\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_bytes\":%d,\"bus_bits\":%d,"
              "\"mem_clock_khz\":%d,\"theoretical_gbs\":%.1f}\n",
              prop.name, cc_major, cc_minor, sms, l2, bus_bits, mem_khz, theoretical_gbs);

  size_t max_bytes = 0;
  for (size_t m : sizes_mib) max_bytes = std::max(max_bytes, m << 20);
  for (size_t m : graph_mib) max_bytes = std::max(max_bytes, (m << 20) * launches_per_graph);
  size_t free_b = 0, total_b = 0;
  CHECK(cudaMemGetInfo(&free_b, &total_b));
  const size_t flush_bytes = std::max<size_t>(size_t(4) * l2, size_t(64) << 20);
  if (2 * max_bytes + flush_bytes > free_b * 9 / 10) {
    std::fprintf(stderr, "not enough free GPU memory for %zu MiB buffers\n", max_bytes >> 20);
    return 2;
  }
  uint4 *a, *b;
  unsigned* sink;
  float* flush;
  CHECK(cudaMalloc(&a, max_bytes));
  CHECK(cudaMalloc(&b, max_bytes));
  CHECK(cudaMalloc(&sink, 4));
  CHECK(cudaMalloc(&flush, flush_bytes));
  CHECK(cudaMemset(a, 1, max_bytes));
  CHECK(cudaMemset(b, 0, max_bytes));
  CHECK(cudaMemset(flush, 0, flush_bytes));
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));
  auto flush_l2 = [&]() { flush_kernel<<<sms * 4, 512, 0, s>>>(flush, flush_bytes / 4); };

  // ---- read and copy over several buffer sizes
  for (size_t mib : sizes_mib) {
    const size_t bytes = mib << 20;
    const size_t n16 = bytes / 16;
    for (int blocks_per_sm : {4, 8}) {
      std::vector<float> t_read, t_copy;
      for (int it = 0; it < iters + 3; ++it) {  // first 3 iterations are warm-up
        float ms = 0;
        flush_l2();
        CHECK(cudaEventRecord(e0, s));
        read_kernel<4><<<sms * blocks_per_sm, 512, 0, s>>>(a, n16, sink, 0);
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 3) t_read.push_back(ms);
        flush_l2();
        CHECK(cudaEventRecord(e0, s));
        copy_kernel<<<sms * blocks_per_sm, 512, 0, s>>>(a, b, n16 / 2);
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 3) t_copy.push_back(ms);
      }
      const float mr = median(t_read), mc = median(t_copy);
      const float br = *std::min_element(t_read.begin(), t_read.end());
      std::printf("{\"test\":\"read\",\"bytes\":%zu,\"blocks_per_sm\":%d,\"iters\":%d,\"ms_median\":%.5f,"
                  "\"gbs\":%.1f,\"gbs_best\":%.1f}\n",
                  bytes, blocks_per_sm, iters, mr, bytes / mr / 1e6, bytes / br / 1e6);
      std::printf("{\"test\":\"copy\",\"bytes\":%zu,\"blocks_per_sm\":%d,\"iters\":%d,\"ms_median\":%.5f,"
                  "\"gbs\":%.1f}\n",
                  bytes, blocks_per_sm, iters, mc, bytes / mc / 1e6);
    }
  }

  // ---- launch-sized reads inside a CUDA graph
#if CUDART_VERSION >= 12030  // PDL edges inside captured graphs need CUDA 12.3+
  const bool pdl_ok = cc_major >= 9;
#else
  const bool pdl_ok = false;
#endif
  for (size_t mib : graph_mib) {
    const size_t bytes = mib << 20;
    for (int pdl = 0; pdl <= (pdl_ok ? 1 : 0); ++pdl) {
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for (int l = 0; l < launches_per_graph; ++l) {
        const uint4* src = a + (size_t)l * (bytes / 16);  // each launch reads its own region
#if CUDART_VERSION >= 11080
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(sms * 2);
        cfg.blockDim = dim3(512);
        cfg.stream = s;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl ? 1 : 0;
        CHECK(cudaLaunchKernelEx(&cfg, read_kernel<1>, src, bytes / 16, sink, pdl));
#else
        read_kernel<1><<<sms * 2, 512, 0, s>>>(src, bytes / 16, sink, 0);
#endif
      }
      CHECK(cudaStreamEndCapture(s, &graph));
      CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
      std::vector<float> t;
      for (int it = 0; it < iters + 3; ++it) {
        float ms = 0;
        flush_l2();
        CHECK(cudaEventRecord(e0, s));
        CHECK(cudaGraphLaunch(exec, s));
        CHECK(cudaEventRecord(e1, s));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= 3) t.push_back(ms);
      }
      const float ms = median(t);
      const double total = (double)bytes * launches_per_graph;
      std::printf("{\"test\":\"graph_read\",\"bytes_per_launch\":%zu,\"launches\":%d,\"pdl\":%d,\"iters\":%d,"
                  "\"us_per_launch\":%.2f,\"gbs\":%.1f}\n",
                  bytes, launches_per_graph, pdl, iters, ms * 1e3 / launches_per_graph, total / ms / 1e6);
      CHECK(cudaGraphExecDestroy(exec));
      CHECK(cudaGraphDestroy(graph));
    }
  }
  CHECK(cudaFree(a));
  CHECK(cudaFree(b));
  CHECK(cudaFree(sink));
  CHECK(cudaFree(flush));
  return 0;
}
