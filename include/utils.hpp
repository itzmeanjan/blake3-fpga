#pragma once
#include "blake3.hpp"

// Executes BLAKE3 kernels with same input size `itr_cnt` -many times and
// computes average execution time of following SYCL commands
//
// - host -> device input tx time
// - kernel execution time
// - device -> host input tx time
void
avg_kernel_exec_tm(sycl::queue& q,
                   size_t chunk_count,
                   size_t itr_cnt,
                   double* const ts)
{
  constexpr size_t ts_size = sizeof(sycl::cl_ulong) * 3;

  // allocate memory on host ( for keeping exec time of enqueued commands )
  sycl::cl_ulong* ts_sum = static_cast<sycl::cl_ulong*>(std::malloc(ts_size));
  sycl::cl_ulong* ts_rnd = static_cast<sycl::cl_ulong*>(std::malloc(ts_size));

  // so that average execution/ data transfer time can be safely computed !
  std::memset(ts_sum, 0, ts_size);

  for (size_t i = 0; i < itr_cnt; i++) {
    const size_t i_size = chunk_count * blake3::CHUNK_LEN;
    constexpr size_t o_size = blake3::OUT_LEN;

    sycl::uchar* i_h = static_cast<sycl::uchar*>(std::malloc(i_size));
    sycl::uchar* i_d =
      static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
    sycl::uchar* o_h = static_cast<sycl::uchar*>(std::malloc(o_size));
    sycl::uchar* o_d =
      static_cast<sycl::uchar*>(sycl::malloc_device(o_size, q));

    // so input is 0xff< --- (i_size - 2) -many `ff` --- >ff
    memset(i_h, 0xff, i_size);

    // host to device input data tx
    sycl::event evt_0 = q.memcpy(i_d, i_h, i_size);
    evt_0.wait();

    // compute on accelerator, wait until completed
    sycl::cl_ulong ts = 0; // exec time of kernels
    blake3::hash(q, i_d, i_size, chunk_count, o_d, &ts);

    // device to host digest tx
    sycl::event evt_1 = q.memcpy(o_h, o_d, blake3::OUT_LEN);
    evt_1.wait();

    ts_sum[0] += time_event(evt_0);
    ts_sum[1] += ts;
    ts_sum[2] += time_event(evt_1);

    std::free(i_h);
    std::free(o_h);
    // because following two allocations are managed by SYCL runtime
    sycl::free(i_d, q);
    sycl::free(o_d, q);
  }

  for (size_t i = 0; i < 3; i++) {
    *(ts + i) = (double)*(ts_sum + i) / (double)itr_cnt;
  }

  // deallocate resources
  std::free(ts_sum);
  std::free(ts_rnd);
}

// Convert nanosecond granularity execution time to readable string i.e. in
// terms of seconds/ milliseconds/ microseconds/ nanoseconds
std::string
to_readable_timespan(double ts)
{
  return ts >= 1e9 ? std::to_string(ts * 1e-9) + " s"
                   : ts >= 1e6 ? std::to_string(ts * 1e-6) + " ms"
                               : ts >= 1e3 ? std::to_string(ts * 1e-3) + " us"
                                           : std::to_string(ts) + " ns";
}
