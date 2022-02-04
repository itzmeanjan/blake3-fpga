#pragma once
#include <CL/sycl.hpp>

// Computes actual execution time of enqueued command with nanosecond level of
// granularity
//
// Ensure that SYCL queue has profiling enabled !
static inline sycl::cl_ulong
time_event(sycl::event& evt)
{
  const sycl::cl_ulong start =
    evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  const sycl::cl_ulong end =
    evt.get_profiling_info<sycl::info::event_profiling::command_end>();

  return end - start;
}
