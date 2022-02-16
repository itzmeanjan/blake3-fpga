#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include "common.hpp"
#include <cassert>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace blake3 {

// just to avoid kernel name mangling in optimization report
class kernelBlake3Hash;

// Following BLAKE3 constants taken from
// https://github.com/itzmeanjan/blake3/blob/1c58f6a343baee52ba1fe7fc98bfb280b6d567da/include/blake3_consts.hpp
constexpr size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                         1, 11, 12, 5,  9, 14, 15, 8 };

// Initial hash values
constexpr uint32_t IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constexpr size_t CHUNK_LEN = 1024; // bytes
constexpr size_t OUT_LEN = 32;     // bytes
constexpr uint32_t BLOCK_LEN = 64; // bytes

// Mixing rounds
constexpr size_t ROUNDS = 7;

// BLAKE3 flags
constexpr uint32_t CHUNK_START = 1 << 0;
constexpr uint32_t CHUNK_END = 1 << 1;
constexpr uint32_t PARENT = 1 << 2;
constexpr uint32_t ROOT = 1 << 3;

// Binary logarithm of n, when n = 2 ^ i | i = {1, 2, 3, ...}
const size_t
bin_log(size_t n)
{
  size_t cnt = 0ul;

  while (n > 1ul) {
    n >>= 1;
    cnt++;
  }

  return cnt;
}

// Compile time check for circular right shift bit position x,
// to ensure that x >= 0 && x < 32
static constexpr bool
is_valid_rot_pos(const uint8_t x)
{
  return x < 32;
}

// Circular right shift of blake3 word `n` ( 32 -bit ) by `x` bit places
// where it's compile time checked that x >= 0 && x < 32
template<uint8_t x>
static const uint32_t
rotr(const uint32_t n) requires(is_valid_rot_pos(x))
{
  return (n >> x) | (n << (32 - x));
}

inline void
g(sycl::private_ptr<uint32_t> state,
  const size_t a,
  const size_t b,
  const size_t c,
  const size_t d,
  const uint32_t mx,
  const uint32_t my)
{
  state[a] = state[a] + state[b] + mx;
  state[d] = rotr<16>(state[d] ^ state[a]);
  state[c] = state[c] + state[d];
  state[b] = rotr<12>(state[b] ^ state[c]);
  state[a] = state[a] + state[b] + my;
  state[d] = rotr<8>(state[d] ^ state[a]);
  state[c] = state[c] + state[d];
  state[b] = rotr<7>(state[b] ^ state[c]);
}

inline void
round(sycl::private_ptr<uint32_t> state, sycl::private_ptr<uint32_t> msg)
{
  // column-wise mixing message words into hash state
  g(state, 0, 4, 8, 12, msg[0], msg[1]);
  g(state, 1, 5, 9, 13, msg[2], msg[3]);
  g(state, 2, 6, 10, 14, msg[4], msg[5]);
  g(state, 3, 7, 11, 15, msg[6], msg[7]);

  // diagonal mixing message words into hash state
  g(state, 0, 5, 10, 15, msg[8], msg[9]);
  g(state, 1, 6, 11, 12, msg[10], msg[11]);
  g(state, 2, 7, 8, 13, msg[12], msg[13]);
  g(state, 3, 4, 9, 14, msg[14], msg[15]);
}

// Permute sixteen BLAKE3 message words of each 64 -bytes wide block, after
// each round of mixing
//
// This routine to be invoked six times ( after each round of mixing, except
// last one, because doing that is redundant ) from following `compress( ... )`
// function
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1623-L1639
inline void
permute(sycl::private_ptr<uint32_t> msg)
{
  // additional on-chip block RAM based memory ( = 64 -bytes ) for helping in
  // message word permutation
  [[intel::fpga_memory("BLOCK_RAM"),
    intel::numbanks(16),
    intel::bankwidth(4)]] uint32_t permuted[16];

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = msg[MSG_PERMUTATION[i]];
  }

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    msg[i] = permuted[i];
  }
}

void
compress(sycl::private_ptr<uint32_t> state, sycl::private_ptr<uint32_t> msg)
{
  round(state, msg);
  permute(msg);

  round(state, msg);
  permute(msg);

  round(state, msg);
  permute(msg);

  round(state, msg);
  permute(msg);

  round(state, msg);
  permute(msg);

  round(state, msg);
  permute(msg);

  round(state, msg);

#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[8 + i];
  }
}

// Four consecutive little endian bytes are interpreted as 32 -bit unsigned
// integer i.e. BLAKE3 message word
static inline const uint32_t
word_from_le_bytes(const sycl::device_ptr<sycl::uchar> input)
{
  return static_cast<uint32_t>(input[3]) << 24 |
         static_cast<uint32_t>(input[2]) << 16 |
         static_cast<uint32_t>(input[1]) << 8 |
         static_cast<uint32_t>(input[0]) << 0;
}

// One 32 -bit BLAKE3 word is converted to four consecutive little endian bytes
inline void
word_to_le_bytes(const uint32_t word, sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    output[i] = static_cast<sycl::uchar>((word >> (i << 3)) & 0xff);
  }
}

// Eight consecutive BLAKE3 message words are converted to 32 little endian
// bytes
void
words_to_le_bytes(const sycl::private_ptr<uint32_t> msg_words,
                  sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    word_to_le_bytes(msg_words[i], output + (i << 2));
  }
}

// BLAKE3 hash function, can be used when chunk count is power of 2
//
// Note, chunk count is preferred to be relatively large number ( say >= 2^20 )
// because this function is supposed to be executed on accelerator i.e. FPGA
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1876-L2006
void
hash(sycl::queue& q,                       // SYCL compute queue
     sycl::uchar* const __restrict input,  // it'll never be modified !
     const size_t i_size,                  // bytes
     const size_t chunk_count,             // works only with power of 2
     sycl::uchar* const __restrict digest, // 32 -bytes BLAKE3 digest
     sycl::cl_ulong* const __restrict ts   // kernel exec time in `ns`
)
{
  // whole input byte array is splitted into N -many chunks, each
  // of 1024 -bytes width
  assert(i_size == chunk_count * CHUNK_LEN);
  // minimum 1MB input size for this implementation
  assert(chunk_count >= (1 << 10)); // but you would probably want >= 2^20
  assert((chunk_count & (chunk_count - 1)) == 0); // ensure power of 2

  const size_t mem_size = (chunk_count * OUT_LEN) << 1;
  uint32_t* mem = static_cast<uint32_t*>(sycl::malloc_device(mem_size, q));

  sycl::event evt =
    q.single_task<kernelBlake3Hash>([=]() [[intel::kernel_args_restrict]] {
      sycl::device_ptr<sycl::uchar> i_ptr{ input };
      sycl::device_ptr<uint32_t> mem_ptr{ mem };
      sycl::device_ptr<sycl::uchar> o_ptr{ digest };

      [[intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(16),
        intel::bankwidth(4)]] uint32_t msg[16];
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(16),
        intel::bankwidth(4)]] uint32_t state[16];

      sycl::private_ptr<uint32_t> state_ptr{ state };
      sycl::private_ptr<uint32_t> msg_ptr{ msg };

      const size_t o_offset = chunk_count << 3;

      const size_t msg_blk_cnt = chunk_count << 4;
      size_t chunk_idx = 0;
      size_t msg_blk_idx = 0;

      [[intel::ivdep]] for (size_t c = 0; c < msg_blk_cnt; c++)
      {
        const size_t i_offset_0 = (chunk_idx << 10) + (msg_blk_idx << 4);
        const size_t o_offset_0 = o_offset + (chunk_idx << 3);

        if (msg_blk_idx == 0) {
#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_ptr[i] = IV[i];
          }
        } else {
#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_ptr[i] = mem_ptr[o_offset_0 + i];
          }
        }

#pragma unroll 4
        for (size_t i = 0; i < 4; i++) {
          state_ptr[8 + i] = IV[i];
        }

        state_ptr[12] = static_cast<uint32_t>(chunk_idx & 0xffffffff);
        state_ptr[13] = static_cast<uint32_t>(chunk_idx >> 32);
        state_ptr[14] = BLOCK_LEN;

        if (msg_blk_idx == 0) {
          state_ptr[15] = CHUNK_START;
        } else if (msg_blk_idx == 15) {
          state_ptr[15] = CHUNK_END;
        } else {
          state_ptr[15] = 0;
        }

#pragma unroll 16
        for (size_t i = 0; i < 16; i++) {
          msg_ptr[i] = word_from_le_bytes(i_ptr + i_offset_0 + (i << 2));
        }

        compress(state_ptr, msg_ptr);

#pragma unroll 8
        for (size_t i = 0; i < 8; i++) {
          mem_ptr[o_offset_0 + i] = state_ptr[i];
        }

        if ((chunk_idx + 1) == chunk_count) {
          chunk_idx = 0;
          msg_blk_idx++;
        } else {
          chunk_idx++;
        }
      }

      const size_t levels = bin_log(chunk_count) - 1;

      for (size_t l = 0; l < levels; l++) {
        const size_t i_offset = (chunk_count << 3) >> l;
        const size_t o_offset = i_offset >> 1;
        const size_t node_cnt = chunk_count >> (l + 1);

        [[intel::ivdep]] for (size_t i = 0; i < node_cnt; i++)
        {
          const size_t i_offset_0 = i_offset + (i << 4);
          const size_t o_offset_0 = o_offset + (i << 3);

#pragma unroll 16
          for (size_t j = 0; j < 16; j++) {
            msg_ptr[j] = mem_ptr[i_offset_0 + j];
          }

#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_ptr[i] = IV[i];
          }
#pragma unroll 4
          for (size_t i = 0; i < 4; i++) {
            state_ptr[8 + i] = IV[i];
          }

          state_ptr[12] = 0;
          state_ptr[13] = 0;
          state_ptr[14] = BLOCK_LEN;
          state_ptr[15] = PARENT;

          compress(state_ptr, msg_ptr);

#pragma unroll 8
          for (size_t j = 0; j < 8; j++) {
            mem_ptr[o_offset_0 + j] = state_ptr[j];
          }
        }
      }

    // computing root chaining value ( blake3 digest )
#pragma unroll 16
      for (size_t j = 0; j < 16; j++) {
        msg_ptr[j] = mem_ptr[16 + j];
      }

#pragma unroll 8
      for (size_t i = 0; i < 8; i++) {
        state_ptr[i] = IV[i];
      }
#pragma unroll 4
      for (size_t i = 0; i < 4; i++) {
        state_ptr[8 + i] = IV[i];
      }

      state_ptr[12] = 0;
      state_ptr[13] = 0;
      state_ptr[14] = BLOCK_LEN;
      state_ptr[15] = PARENT | ROOT;

      compress(state_ptr, msg_ptr);

      // writing little endian byte digest back to desired memory allocation
      words_to_le_bytes(state_ptr, o_ptr);
    });

  evt.wait();
  sycl::free(mem, q);

  if (ts != nullptr) {
    *ts = time_event(evt);
  }
}
}
