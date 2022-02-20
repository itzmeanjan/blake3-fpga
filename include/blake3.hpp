#pragma once
#include "common.hpp"
#include <cassert>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace blake3 {

// Just to avoid kernel name mangling in optimization report
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
static inline const uint32_t
rotr(const uint32_t n) requires(is_valid_rot_pos(x))
{
  return (n >> x) | (n << (32 - x));
}

// BLAKE3 mixing function, which mixes message words ( 64 -bytes ) into hash
// state either column-wise/ diagonally
//
// Taken from
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L42-L52
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

// BLAKE3 round function, which is invoked 7 times for mixing 64 -bytes message
// words into hash state
//
// During each mixing round, a permutation of 64 -bytes message words are mixed
// into hash state both column-wise and diagonally
//
// Taken from
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L54-L65
inline void
round(sycl::private_ptr<uint32_t> state, sycl::private_ptr<uint32_t> msg)
{
  // column-wise mixing of message words into hash state
  g(state, 0, 4, 8, 12, msg[0], msg[1]);
  g(state, 1, 5, 9, 13, msg[2], msg[3]);
  g(state, 2, 6, 10, 14, msg[4], msg[5]);
  g(state, 3, 7, 11, 15, msg[6], msg[7]);

  // diagonal mixing of message words into hash state
  g(state, 0, 5, 10, 15, msg[8], msg[9]);
  g(state, 1, 6, 11, 12, msg[10], msg[11]);
  g(state, 2, 7, 8, 13, msg[12], msg[13]);
  g(state, 3, 4, 9, 14, msg[14], msg[15]);
}

// Permute sixteen BLAKE3 message words of 64 -bytes wide block, after
// each round of mixing
//
// This routine to be invoked six times ( after each round of mixing, except
// last one, because doing that is not necessary ) from following `compress( ...
// )` function
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1623-L1639
inline void
permute(sycl::private_ptr<uint32_t> msg)
{
  // additional FPGA register based memory ( = 64 -bytes ) for helping in
  // message word permutation
  [[intel::fpga_register]] uint32_t permuted[16];

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = msg[MSG_PERMUTATION[i]];
  }

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    msg[i] = permuted[i];
  }
}

// BLAKE3 compression function, which is used for compressing 64 -bytes
// input message block ( 16 words ) into 32 -bytes output chaining value ( 8
// words )
void
compress(sycl::private_ptr<uint32_t> state, sycl::private_ptr<uint32_t> msg)
{
  // round 1
  round(state, msg);
  permute(msg);

  // round 2
  round(state, msg);
  permute(msg);

  // round 3
  round(state, msg);
  permute(msg);

  // round 4
  round(state, msg);
  permute(msg);

  // round 5
  round(state, msg);
  permute(msg);

  // round 6
  round(state, msg);
  permute(msg);

  // round 7
  round(state, msg);

  // prepare output chaining value of this message block compression
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[8 + i];
    // note, that
    // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
    // can be skipped, because it doesn't affect what output chaining value will
    // be
    //
    // results in lesser hardware synthesized !
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
// bytes ( used when writing BLAKE3 digest back to global memory )
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
// Note, chunk count is preferred to be relatively large number ( say >= 2^10 )
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

  // temporary memory allocation on global memory for keeping all intermediate
  // chaining values
  //
  // note, chaining values are organized as nodes are kept in fully computed (
  // all intermediate nodes, along with leaves ) binary merkle tree
  //
  // @todo this allocation size can be improved !
  const size_t mem_size = (chunk_count * OUT_LEN) << 1;
  uint32_t* mem = static_cast<uint32_t*>(sycl::malloc_device(mem_size, q));

  sycl::event evt =
    q.single_task<kernelBlake3Hash>([=]() [[intel::kernel_args_restrict]] {
      // Just to hint that Load Store Units don't need to interface with host
      sycl::device_ptr<sycl::uchar> i_ptr{ input };
      sycl::device_ptr<uint32_t> mem_ptr{ mem };
      sycl::device_ptr<sycl::uchar> o_ptr{ digest };

      // on-chip FPGA register based allocation where input message words ( 64
      // -bytes ) and hash state (64 -bytes ) are kept
      //
      // FPGA registers are kind of abundant ( in this context ), and they allow
      // stall-free access to each of 16 elements of these arrays --- so should
      // yield better performance !
      [[intel::fpga_register]] uint32_t msg_0[16];
      [[intel::fpga_register]] uint32_t state_0[16];

      // just to hint that these FPGA register backed array allocations
      // are kept on private memory ( read private to single work-item )
      sycl::private_ptr<uint32_t> state_0_ptr{ state_0 };
      sycl::private_ptr<uint32_t> msg_0_ptr{ msg_0 };

      const size_t o_offset = chunk_count << 3;

      // for compressing all chunks, these many compress( ... ) function
      // calls need to be performed, because each chunk has 16 message blocks
      // each of length 64 -bytes, making total of 1024 -bytes wide chunk
      const size_t msg_blk_cnt = chunk_count << 4;
      size_t chunk_idx = 0;
      size_t msg_blk_idx = 0;

      // --- chunk compression section ---
      //
      // each chunk has 16 message blocks, which are compressed sequentially
      // due to input/ output chaining value dependency
      //
      // in following for loop i-th chunk's j-th message block is compressed
      // first, resulting chaining value is written back to global memory (
      // expensive op ! )
      //
      // after that (i + 1)-th chunk's j-th message block is compressed and
      // resulting chaining value is written to next 32 -bytes memory on SYCL
      // global memory ( using `mem_ptr` )
      //
      // this keeps going on, until all N -many chunk's j-th message blocks are
      // processed
      //
      // until now, j = 0
      //
      // now j = 1
      //
      // and we start by processing i-th chunk's j-th block and keep doing until
      // all chunk's second message blocks are compressed, while using j = 0's
      // output chaining values ( computed when j = 0 ) as input chaining values
      // for each chunk
      //
      // finally, j = 15 i.e. last message block of each chunk
      //
      // for j = 15, all chunk's message blocks are compressed and it produces N
      // -many output chaining values for N -many chunks, which were compressed
      // in 16 consecutive rounds
      [[intel::ivdep]] for (size_t c = 0; c < msg_blk_cnt; c++)
      {
        const size_t i_offset_0 = (chunk_idx << 10) + (msg_blk_idx << 4);
        const size_t o_offset_0 = o_offset + (chunk_idx << 3);

        // for first message block of each chunk, input chaining values are
        // constant initial hash values
        if (msg_blk_idx == 0) {
#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_0_ptr[i] = IV[i];
          }
        } else {
        // for all remaining message blocks input chaining values are
        // output chaining values obtained by compressing previous message block
#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_0_ptr[i] = mem_ptr[o_offset_0 + i];
          }
        }

      // prepare hash state, see
      // https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1649-L1657
      // to understand how hash state is prepared
      //
      // or you may want to see non-SIMD implementation
      // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L82-L99
#pragma unroll 4
        for (size_t i = 0; i < 4; i++) {
          state_0_ptr[8 + i] = IV[i];
        }

        state_0_ptr[12] = static_cast<uint32_t>(chunk_idx & 0xffffffff);
        state_0_ptr[13] = static_cast<uint32_t>(chunk_idx >> 32);
        state_0_ptr[14] = BLOCK_LEN;

        if (msg_blk_idx == 0) {
          state_0_ptr[15] = CHUNK_START;
        } else if (msg_blk_idx == 15) {
          state_0_ptr[15] = CHUNK_END;
        } else {
          state_0_ptr[15] = 0;
        }

      // 64 -bytes message block read from global memory ( expensive, but
      // nothing much to do to avoid this ! )
#pragma unroll 16
        for (size_t i = 0; i < 16; i++) {
          msg_0_ptr[i] = word_from_le_bytes(i_ptr + i_offset_0 + (i << 2));
        }

        // compress four message block(s) from four consecutive chunks
        compress(state_0_ptr, msg_0_ptr);

      // obtain 32 -bytes output chaining values, and write back to global
      // memory ( expensive, can attempt to avoid this ! )
#pragma unroll 8
        for (size_t i = 0; i < 8; i++) {
          mem_ptr[o_offset_0 + i] = state_0_ptr[i];
        }

        // point to next chunk/ message block
        if ((chunk_idx + 1) == chunk_count) {
          chunk_idx = 0;
          msg_blk_idx++;
        } else {
          chunk_idx++;
        }
      }
      //
      // --- chunk compression ---

      // --- parent chaining value computation using binary merklization ---
      //
      // except root ( chaining values ) of BLAKE3 merkle tree, all intermediate
      // parent chaining values are to be computed in data-dependent `levels`
      // -many rounds
      const size_t levels = bin_log(chunk_count) - 1;

      // level (i + 1) consumes level i as input ( where leaf nodes are already
      // computed, see above chunk compression section )
      for (size_t l = 0; l < levels; l++) {
        const size_t i_offset = (chunk_count << 3) >> l;
        const size_t o_offset = i_offset >> 1;
        const size_t node_cnt = chunk_count >> (l + 1);

        // these many intermediate chaining values are to be computed in this
        // level of BLAKE3 binary merkle tree
        [[intel::ivdep]] for (size_t i = 0; i < node_cnt; i++)
        {
          const size_t i_offset_0 = i_offset + (i << 4);
          const size_t o_offset_0 = o_offset + (i << 3);

        // read 64 -bytes message words from global memory
#pragma unroll 16
          for (size_t j = 0; j < 16; j++) {
            msg_0_ptr[j] = mem_ptr[i_offset_0 + j];
          }

        // input chaining values being placed in first 8 words of hash state
#pragma unroll 8
          for (size_t i = 0; i < 8; i++) {
            state_0_ptr[i] = IV[i];
          }
#pragma unroll 4
          for (size_t i = 0; i < 4; i++) {
            state_0_ptr[8 + i] = IV[i];
          }

          state_0_ptr[12] = 0;
          state_0_ptr[13] = 0;
          state_0_ptr[14] = BLOCK_LEN;
          state_0_ptr[15] = PARENT;

          // compressing two message blocks, living next to each other
          compress(state_0_ptr, msg_0_ptr);

        // producing parent chaining values
#pragma unroll 8
          for (size_t j = 0; j < 8; j++) {
            mem_ptr[o_offset_0 + j] = state_0_ptr[j];
          }
        }
      }
    //
    // --- parent chaining value computation using binary merklization ---

    // --- computing root chaining values ( BLAKE3 digest ) ---
#pragma unroll 16
      for (size_t j = 0; j < 16; j++) {
        msg_0_ptr[j] = mem_ptr[16 + j];
      }

#pragma unroll 8
      for (size_t i = 0; i < 8; i++) {
        state_0_ptr[i] = IV[i];
      }
#pragma unroll 4
      for (size_t i = 0; i < 4; i++) {
        state_0_ptr[8 + i] = IV[i];
      }

      state_0_ptr[12] = 0;
      state_0_ptr[13] = 0;
      state_0_ptr[14] = BLOCK_LEN;
      state_0_ptr[15] = PARENT | ROOT;

      compress(state_0_ptr, msg_0_ptr);
      // --- computing root chaining values ( BLAKE3 digest ) ---

      // writing little endian digest bytes back to desired memory allocation
      words_to_le_bytes(state_0_ptr, o_ptr);
    });

  evt.wait();
  sycl::free(mem, q);

  if (ts != nullptr) {
    *ts = time_event(evt);
  }
}
}
