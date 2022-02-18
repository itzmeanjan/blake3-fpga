#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
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

// BLAKE3 Hash State ( four 128 -bit vectors ) lane-wise rightwards rotation
// factors
constexpr sycl::uint4 rrot_16 = sycl::uint4(16); // = 32 - 16
constexpr sycl::uint4 rrot_12 = sycl::uint4(20); // = 32 - 12
constexpr sycl::uint4 rrot_8 = sycl::uint4(24);  // = 32 - 8
constexpr sycl::uint4 rrot_7 = sycl::uint4(25);  // = 32 - 7

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

// BLAKE3 round function, which is invoked 7 times for mixing 64 -bytes message
// words into hash state, which is represented using four 128 -bit vectors
//
// During each mixing round, a permutation of 64 -bytes message words are mixed
// into hash state both column-wise and diagonally
//
// Taken from
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1569-L1621
inline void
round(sycl::private_ptr<sycl::uint4> state, sycl::private_ptr<uint32_t> msg)
{
  const sycl::uint4 mx = sycl::uint4(msg[0], msg[2], msg[4], msg[6]);
  const sycl::uint4 my = sycl::uint4(msg[1], msg[3], msg[5], msg[7]);
  const sycl::uint4 mz = sycl::uint4(msg[8], msg[10], msg[12], msg[14]);
  const sycl::uint4 mw = sycl::uint4(msg[9], msg[11], msg[13], msg[15]);

  // column-wise mixing
  state[0] = state[0] + state[1] + mx;
  state[3] = sycl::rotate(state[3] ^ state[0], rrot_16);
  state[2] = state[2] + state[3];
  state[1] = sycl::rotate(state[1] ^ state[2], rrot_12);
  state[0] = state[0] + state[1] + my;
  state[3] = sycl::rotate(state[3] ^ state[0], rrot_8);
  state[2] = state[2] + state[3];
  state[1] = sycl::rotate(state[1] ^ state[2], rrot_7);

  // diagonalize
  state[1] = state[1].yzwx();
  state[2] = state[2].zwxy();
  state[3] = state[3].wxyz();

  // diagonal mixing
  state[0] = state[0] + state[1] + mz;
  state[3] = sycl::rotate(state[3] ^ state[0], rrot_16);
  state[2] = state[2] + state[3];
  state[1] = sycl::rotate(state[1] ^ state[2], rrot_12);
  state[0] = state[0] + state[1] + mw;
  state[3] = sycl::rotate(state[3] ^ state[0], rrot_8);
  state[2] = state[2] + state[3];
  state[1] = sycl::rotate(state[1] ^ state[2], rrot_7);

  // un-diagonalize
  state[1] = state[1].wxyz();
  state[2] = state[2].zwxy();
  state[3] = state[3].yzwx();
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
compress(sycl::private_ptr<sycl::uint4> state, sycl::private_ptr<uint32_t> msg)
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
  state[0] ^= state[2];
  state[1] ^= state[3];
  // note, that
  // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
  // can be skipped, because it doesn't affect what output chaining value will
  // be
  //
  // results in lesser hardware synthesized !
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
words_to_le_bytes(const sycl::device_ptr<uint32_t> msg_words,
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
  // note, chaining values are organized as nodes are kept in computed ( all
  // intermediate nodes ) binary merkle tree
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
      [[intel::fpga_register]] sycl::uint4 state_0[4];
      [[intel::fpga_register]] uint32_t msg_1[16];
      [[intel::fpga_register]] sycl::uint4 state_1[4];

      // just to hint that these FPGA register backed array allocations
      // are kept on private memory ( read private to single work-item )
      sycl::private_ptr<sycl::uint4> state_0_ptr{ state_0 };
      sycl::private_ptr<uint32_t> msg_0_ptr{ msg_0 };
      sycl::private_ptr<sycl::uint4> state_1_ptr{ state_1 };
      sycl::private_ptr<uint32_t> msg_1_ptr{ msg_1 };

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
      [[intel::ivdep]] for (size_t c = 0; c < msg_blk_cnt; c += 2)
      {
        const size_t i_offset_0 = (chunk_idx << 10) + (msg_blk_idx << 4);
        const size_t i_offset_1 = ((chunk_idx + 1) << 10) + (msg_blk_idx << 4);
        const size_t o_offset_0 = o_offset + (chunk_idx << 3);
        const size_t o_offset_1 = o_offset + ((chunk_idx + 1) << 3);

        // for first message block of each chunk, input chaining values are
        // constant initial hash values
        if (msg_blk_idx == 0) {
          state_0_ptr[0] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_0_ptr[1] = sycl::uint4(IV[4], IV[5], IV[6], IV[7]);

          state_1_ptr[0] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_1_ptr[1] = sycl::uint4(IV[4], IV[5], IV[6], IV[7]);
        } else {
          // for all remaining message blocks input chaining values are
          // output chaining values obtained by compressing previous message
          // block
          state_0_ptr[0] = sycl::uint4(mem_ptr[o_offset_0 + 0],
                                       mem_ptr[o_offset_0 + 1],
                                       mem_ptr[o_offset_0 + 2],
                                       mem_ptr[o_offset_0 + 3]);
          state_0_ptr[1] = sycl::uint4(mem_ptr[o_offset_0 + 4],
                                       mem_ptr[o_offset_0 + 5],
                                       mem_ptr[o_offset_0 + 6],
                                       mem_ptr[o_offset_0 + 7]);

          state_1_ptr[0] = sycl::uint4(mem_ptr[o_offset_1 + 0],
                                       mem_ptr[o_offset_1 + 1],
                                       mem_ptr[o_offset_1 + 2],
                                       mem_ptr[o_offset_1 + 3]);
          state_1_ptr[1] = sycl::uint4(mem_ptr[o_offset_1 + 4],
                                       mem_ptr[o_offset_1 + 5],
                                       mem_ptr[o_offset_1 + 6],
                                       mem_ptr[o_offset_1 + 7]);
        }

        // prepare hash state, see
        // https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1649-L1657
        // to understand how hash state is prepared
        //
        // or you may want to see non-SIMD implementation
        // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L82-L99
        state_0_ptr[2] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
        state_1_ptr[2] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);

        if (msg_blk_idx == 0) {
          state_0_ptr[3] =
            sycl::uint4(static_cast<uint32_t>(chunk_idx & 0xffffffff),
                        static_cast<uint32_t>(chunk_idx >> 32),
                        BLOCK_LEN,
                        CHUNK_START);
          state_1_ptr[3] =
            sycl::uint4(static_cast<uint32_t>((chunk_idx + 1) & 0xffffffff),
                        static_cast<uint32_t>((chunk_idx + 1) >> 32),
                        BLOCK_LEN,
                        CHUNK_START);
        } else if (msg_blk_idx == 15) {
          state_0_ptr[3] =
            sycl::uint4(static_cast<uint32_t>(chunk_idx & 0xffffffff),
                        static_cast<uint32_t>(chunk_idx >> 32),
                        BLOCK_LEN,
                        CHUNK_END);
          state_1_ptr[3] =
            sycl::uint4(static_cast<uint32_t>((chunk_idx + 1) & 0xffffffff),
                        static_cast<uint32_t>((chunk_idx + 1) >> 32),
                        BLOCK_LEN,
                        CHUNK_END);
        } else {
          state_0_ptr[3] =
            sycl::uint4(static_cast<uint32_t>(chunk_idx & 0xffffffff),
                        static_cast<uint32_t>(chunk_idx >> 32),
                        BLOCK_LEN,
                        0);
          state_1_ptr[3] =
            sycl::uint4(static_cast<uint32_t>((chunk_idx + 1) & 0xffffffff),
                        static_cast<uint32_t>((chunk_idx + 1) >> 32),
                        BLOCK_LEN,
                        0);
        }

      // 64 -bytes message block read from global memory ( expensive, but
      // nothing much to do to avoid this ! )
#pragma unroll 16
        for (size_t i = 0; i < 16; i++) {
          msg_0_ptr[i] = word_from_le_bytes(i_ptr + i_offset_0 + (i << 2));
        }
#pragma unroll 16
        for (size_t i = 0; i < 16; i++) {
          msg_1_ptr[i] = word_from_le_bytes(i_ptr + i_offset_1 + (i << 2));
        }

        // compress this message block
        compress(state_0_ptr, msg_0_ptr);
        compress(state_1_ptr, msg_1_ptr);

      // obtain 32 -bytes output chaining values, and write back to global
      // memory ( expensive, can attempt to avoid this ! )
#pragma unroll 2
        for (size_t i = 0; i < 2; i++) {
          mem_ptr[o_offset_0 + (i << 2) + 0] = state_0_ptr[i].x();
          mem_ptr[o_offset_0 + (i << 2) + 1] = state_0_ptr[i].y();
          mem_ptr[o_offset_0 + (i << 2) + 2] = state_0_ptr[i].z();
          mem_ptr[o_offset_0 + (i << 2) + 3] = state_0_ptr[i].w();

          mem_ptr[o_offset_1 + (i << 2) + 0] = state_1_ptr[i].x();
          mem_ptr[o_offset_1 + (i << 2) + 1] = state_1_ptr[i].y();
          mem_ptr[o_offset_1 + (i << 2) + 2] = state_1_ptr[i].z();
          mem_ptr[o_offset_1 + (i << 2) + 3] = state_1_ptr[i].w();
        }

        // point to next chunk/ message block
        if ((chunk_idx + 2) == chunk_count) {
          chunk_idx = 0;
          msg_blk_idx++;
        } else {
          chunk_idx += 2;
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
        [[intel::ivdep]] for (size_t i = 0; i < node_cnt; i += 2)
        {
          const size_t i_offset_0 = i_offset + (i << 4);
          const size_t i_offset_1 = i_offset + ((i + 1) << 4);
          const size_t o_offset_0 = o_offset + (i << 3);
          const size_t o_offset_1 = o_offset + ((i + 1) << 3);

        // read 64 -bytes message words from global memory
#pragma unroll 16
          for (size_t j = 0; j < 16; j++) {
            msg_0_ptr[j] = mem_ptr[i_offset_0 + j];
          }
#pragma unroll 16
          for (size_t j = 0; j < 16; j++) {
            msg_1_ptr[j] = mem_ptr[i_offset_1 + j];
          }

          // input chaining values being placed in first 8 words of hash state
          state_0_ptr[0] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_0_ptr[1] = sycl::uint4(IV[4], IV[5], IV[6], IV[7]);
          state_0_ptr[2] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_0_ptr[3] = sycl::uint4(0, 0, BLOCK_LEN, PARENT);

          state_1_ptr[0] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_1_ptr[1] = sycl::uint4(IV[4], IV[5], IV[6], IV[7]);
          state_1_ptr[2] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
          state_1_ptr[3] = sycl::uint4(0, 0, BLOCK_LEN, PARENT);

          // compressing this message block
          compress(state_0_ptr, msg_0_ptr);
          compress(state_1_ptr, msg_1_ptr);

          // producing parent chaining value
          mem_ptr[o_offset_0 + 0] = state_0_ptr[0].x();
          mem_ptr[o_offset_0 + 1] = state_0_ptr[0].y();
          mem_ptr[o_offset_0 + 2] = state_0_ptr[0].z();
          mem_ptr[o_offset_0 + 3] = state_0_ptr[0].w();
          mem_ptr[o_offset_0 + 4] = state_0_ptr[1].x();
          mem_ptr[o_offset_0 + 5] = state_0_ptr[1].y();
          mem_ptr[o_offset_0 + 6] = state_0_ptr[1].z();
          mem_ptr[o_offset_0 + 7] = state_0_ptr[1].w();

          mem_ptr[o_offset_1 + 0] = state_1_ptr[0].x();
          mem_ptr[o_offset_1 + 1] = state_1_ptr[0].y();
          mem_ptr[o_offset_1 + 2] = state_1_ptr[0].z();
          mem_ptr[o_offset_1 + 3] = state_1_ptr[0].w();
          mem_ptr[o_offset_1 + 4] = state_1_ptr[1].x();
          mem_ptr[o_offset_1 + 5] = state_1_ptr[1].y();
          mem_ptr[o_offset_1 + 6] = state_1_ptr[1].z();
          mem_ptr[o_offset_1 + 7] = state_1_ptr[1].w();
        }
      }
    //
    // --- parent chaining value computation using binary merklization ---

    // --- computing root chaining values ( BLAKE3 digest ) ---
    //
    // prepare input message words
#pragma unroll 16
      for (size_t i = 0; i < 16; i++) {
        msg_0_ptr[i] = mem_ptr[16 + i];
      }

      // prepare hash state
      state_0_ptr[0] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
      state_0_ptr[1] = sycl::uint4(IV[4], IV[5], IV[6], IV[7]);
      state_0_ptr[2] = sycl::uint4(IV[0], IV[1], IV[2], IV[3]);
      state_0_ptr[3] = sycl::uint4(0, 0, BLOCK_LEN, PARENT | ROOT);

      // compress and find root of blake3 merkle tree
      compress(state_0_ptr, msg_0_ptr);

    // write output chaining values back to global memory
#pragma unroll 2
      for (size_t i = 0; i < 2; i++) {
        mem_ptr[8 + (i << 2) + 0] = state_0_ptr[i].x();
        mem_ptr[8 + (i << 2) + 1] = state_0_ptr[i].y();
        mem_ptr[8 + (i << 2) + 2] = state_0_ptr[i].z();
        mem_ptr[8 + (i << 2) + 3] = state_0_ptr[i].w();
      }
      // --- computing root chaining values ( BLAKE3 digest ) ---

      // writing little endian digest bytes back to desired memory allocation
      words_to_le_bytes(mem_ptr + 8, o_ptr);
    });

  evt.wait();
  sycl::free(mem, q);

  if (ts != nullptr) {
    *ts = time_event(evt);
  }
}
}
