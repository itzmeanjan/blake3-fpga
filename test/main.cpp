#include "blake3.hpp"
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
  sycl::context c{ d };
  sycl::queue q{ c, d };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  constexpr size_t chunk_count = 1 << 10;

  // in bash console
  //
  // $ python3 -m pip install --user blake3
  //
  // in python3 console
  //
  // >>> import blake3
  // >>> a = [0xff] * (1 << 20)
  // >>> list(blake3.blake3(bytes(a)).digest())
  constexpr sycl::uchar expected[32] = {
    3,   107, 169, 54, 188, 220, 105, 198, 56,  19, 158,
    182, 125, 203, 4,  77,  220, 197, 132, 215, 44, 187,
    125, 130, 161, 92, 234, 112, 223, 45,  212, 205
  };

  constexpr size_t i_size = chunk_count * blake3::CHUNK_LEN;
  constexpr size_t o_size = blake3::OUT_LEN;

  sycl::uchar* i_h = static_cast<sycl::uchar*>(malloc(i_size));
  sycl::uchar* i_d = static_cast<sycl::uchar*>(sycl::malloc_device(i_size, q));
  sycl::uchar* o_h = static_cast<sycl::uchar*>(malloc(o_size));
  sycl::uchar* o_d = static_cast<sycl::uchar*>(sycl::malloc_device(o_size, q));

  // so input is 0xff< --- (i_size - 2) -many `ff` --- >ff
  memset(i_h, 0xff, i_size);

  // host to device input data tx
  q.memcpy(i_d, i_h, i_size).wait();
  // compute on accelerator, wait until completed
  blake3::hash(q, i_d, i_size, chunk_count, o_d, nullptr);
  // device to host digest tx
  q.memcpy(o_h, o_d, blake3::OUT_LEN).wait();

  for (size_t i = 0; i < blake3::OUT_LEN; i++) {
    assert(o_h[i] == expected[i]);
  }

  // managed by SYCL runtime
  sycl::free(i_d, q);
  sycl::free(o_d, q);

  std::free(i_h);
  std::free(o_h);

  std::cout << "passed blake3 test !" << std::endl;

  return EXIT_SUCCESS;
}
