#include "utils.hpp"
#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>

#if !(defined FPGA_EMU || defined FPGA_HW)
#define FPGA_EMU
#endif

int
main(int argc, char** argv)
{

#if defined FPGA_EMU
  sycl::ext::intel::fpga_emulator_selector s{};
#elif defined FPGA_HW
  sycl::ext::intel::fpga_selector s{};
#endif

  sycl::device d{ s };
  // using explicit context, instead of relying on default context created by
  // sycl::queue
  sycl::context ctx{ d };
  // enabling profiling in queue is required when benchmarking blake3
  // implementation
  sycl::queue q{ ctx, d, sycl::property::queue::enable_profiling() };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  constexpr size_t itr_cnt = 8;
  double* ts = static_cast<double*>(std::malloc(sizeof(double) * 3));

  std::cout << "Benchmarking BLAKE3 FPGA implementation" << std::endl
            << std::endl;
  std::cout << std::setw(24) << std::right << "input size"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << "\t\t" << std::setw(16) << std::right << "host-to-device tx time"
            << "\t\t" << std::setw(16) << std::right << "device-to-host tx time"
            << std::endl;

  for (size_t i = 1 << 10; i <= 1 << 20; i <<= 1) {
    avg_kernel_exec_tm(q, i, itr_cnt, ts);

    std::cout << std::setw(20) << std::right << ((i * blake3::CHUNK_LEN) >> 20)
              << " MB"
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(*(ts + 1)) << "\t\t" << std::setw(22)
              << std::right << to_readable_timespan(*(ts + 0)) << "\t\t"
              << std::setw(22) << std::right << to_readable_timespan(*(ts + 2))
              << std::endl;
  }

  return EXIT_SUCCESS;
}
