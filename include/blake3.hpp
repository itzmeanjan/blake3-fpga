#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include "common.hpp"
#include <cassert>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace blake3 {

// just to avoid kernel name mangling issue in optimization report
class kernelBlake3Orchestrator;
class kernelBlake3Compressor;

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

// Blake3 state vector rotation factors, during application of 7 blake3 rounds
constexpr sycl::uint4 rrot_16 = sycl::uint4(16); // = 32 - 16
constexpr sycl::uint4 rrot_12 = sycl::uint4(20); // = 32 - 12
constexpr sycl::uint4 rrot_8 = sycl::uint4(24);  // = 32 - 8
constexpr sycl::uint4 rrot_7 = sycl::uint4(25);  // = 32 - 7

// For SYCL pipe, following design document
// https://github.com/intel/llvm/blob/ad9ac98/sycl/doc/extensions/proposed/SYCL_EXT_INTEL_DATAFLOW_PIPES.asciidoc
//
//
// Pipe to be used for sending initial hash state of 64 -bytes to compressor
// kernel from orchestrator kernel
using i_pipe0 = sycl::ext::intel::pipe<class HashStatePipe, uint32_t, 16>;
// Pipe to be used for sending ( total ) 64 -bytes message words to compressor
// kernel from orchestrator kernel
using i_pipe1 = sycl::ext::intel::pipe<class MessageWordsPipe, uint32_t, 16>;
// Pipe to be used for sending 32 -bytes output chaining value as result of
// compression, from compressor to orchestrator kernel
using o_pipe0 = sycl::ext::intel::pipe<class ChainingValuePipe, uint32_t, 8>;

// BLAKE3 round, applied 7 times for mixing sixteen message words ( = total 64
// -bytes ) into hash state, both column-wise and diagonally !
//
// See
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L54-L65
static inline void
rnd(sycl::private_ptr<sycl::uint4> state, sycl::private_ptr<uint32_t> msg)
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
permute(sycl::private_ptr<uint32_t> msg_words)
{
  // additional array memory ( = 64 -bytes ) for helping in message word
  // permutation
  //
  // probably going to be optimized such that it's not synthesized !
  [[intel::fpga_memory("BLOCK_RAM"),
    intel::numbanks(16),
    intel::bankwidth(4)]] uint32_t permuted[16];

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    permuted[i & 0xf] = msg_words[MSG_PERMUTATION[i & 0xf] & 0xf];
  }

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    msg_words[i & 0xf] = permuted[i & 0xf];
  }
}

// BLAKE3 compression function, which mixes 64 -bytes message block into
// sixteen word hash state ( which is also 64 -bytes, because word size of
// BLAKE3 is 32 -bit ) by 7 mixing rounds and 6 permutation rounds ( note, last
// mixing round doesn't require one following permutation round )
//
// Returned output chaining value ( i.e. result of this message block
// compression ) is kept on first 8 words of state array ( = 32 -bytes )
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1641-L1703
static inline void
compress(sycl::private_ptr<sycl::uint4> state, // hash state
         sycl::private_ptr<uint32_t> msg_words // input message
)
{
  // round 1
  rnd(state, msg_words);
  permute(msg_words);

  // round 2
  rnd(state, msg_words);
  permute(msg_words);

  // round 3
  rnd(state, msg_words);
  permute(msg_words);

  // round 4
  rnd(state, msg_words);
  permute(msg_words);

  // round 5
  rnd(state, msg_words);
  permute(msg_words);

  // round 6
  rnd(state, msg_words);
  permute(msg_words);

  // round 7
  rnd(state, msg_words);
  // no need to permute message words anymore !

  // computing output chaining value of message block
  // and keeping it over first 8 words of hash state
  state[0] ^= state[2];
  state[1] ^= state[3];
  // notice I'm skipping
  // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
  // because that statement doesn't dictate what output chaining value of this
  // message block will be !
  //
  // So it's safe to do so !
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
static inline void
word_to_le_bytes(const uint32_t word, sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    output[i] = static_cast<sycl::uchar>((word >> (i << 3)) & 0xff);
  }
}

// Eight consecutive BLAKE3 message words are converted to 32 little endian
// bytes
static inline void
words_to_le_bytes(const sycl::private_ptr<uint32_t> msg_words,
                  sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    word_to_le_bytes(msg_words[i & 0x7], output + (i << 2));
  }
}

// Binary logarithm of n, when n = 2 ^ i | i = {1, 2, ...}
static inline const size_t
bin_log(size_t n)
{
  size_t cnt = 0ul;

  while (n > 1ul) {
    n >>= 1;
    cnt++;
  }

  return cnt;
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

  const size_t mem_size = (chunk_count << 3) * sizeof(uint32_t);
  uint32_t* mem = static_cast<uint32_t*>(sycl::malloc_device(mem_size, q));

  sycl::event evt0 = q.single_task<kernelBlake3Orchestrator>([=
  ]() [[intel::kernel_args_restrict]] {
    // Following best practices guide, so that it hints compiler that following
    // three allocations are on device memory & LSU doesn't need to interface
    // with host memory unnecessarily
    sycl::device_ptr<sycl::uchar> i_ptr = sycl::device_ptr<sycl::uchar>(input);
    sycl::device_ptr<uint32_t> mem_ptr = sycl::device_ptr<uint32_t>(mem);
    // sycl::device_ptr<sycl::uchar> o_ptr = sycl::device_ptr<sycl::uchar>(digest);

    // mem_idx ∈ [0, N) | N = chunk count
    [[intel::fpga_register]] size_t mem_idx = 0ul;

    // temporary storage for input/ output chaining values;
    // used for message passing over on-chip fifo pipes
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::numbanks(8),
      intel::bankwidth(4)]] uint32_t cv0[8];

    // temporary storage for input message words ( = 64 -bytes ),
    // so that burst coalesced 512 -bit access can be performed from memory
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::numbanks(16),
      intel::bankwidth(4)]] uint32_t msg[16];

    [[intel::ivdep]] for (size_t i = 0; i < chunk_count; i += 1)
    {
      // first chunk being worked on
      [[intel::fpga_register]] const size_t chunk_id0 = i;
      // second chunk being worked on
      //
      // note, two chunks are living next to each other,
      // but specific message blocks from those chunks, which are
      // to be compressed in following statements, are *never*
      // contiguous in memory
      // [[intel::fpga_register]] const size_t chunk_id1 = i + 1ul;
      // computing address offset of global memory input byte array
      // for both chunks, which are to be compressed, in following section
      [[intel::fpga_register]] const size_t _i_ptr0 = chunk_id0 << 10;
      // [[intel::fpga_register]] const size_t _i_ptr1 = chunk_id1 << 10;

      // fill input chaining values ( with predefined initial hash values ) for
      // both chunk's first message blocks i.e. message block 0
#pragma unroll 8 // fully parallelized, will be burst coalesced !
      for (size_t j = 0; j < 8; j++) {
        cv0[j & 0x7] = IV[j & 0x7];
        // cv1[j & 0x7] = IV[j & 0x7];
      }

      // each chunk has 1024 -bytes, meaning 16 rounds of compression will be
      // required before we get to compute output chaining value of a chunk
      //
      // in following for loop, I'm compressing (16 + 16) message blocks
      // from two chunks
      for (size_t j = 0; j < 16; j++) {
        // to be compressed message block's beginning, for first chunk
        sycl::device_ptr<sycl::uchar> ptr0 = i_ptr + _i_ptr0 + (j << 6);
        // to be compressed message block's beginning, for second chunk
        // sycl::device_ptr<sycl::uchar> ptr1 = i_ptr + _i_ptr1 + (j << 6);

        // --- begin compressing first chunk's message block ---

#pragma unroll 16 // burst coalesced read of 512 -bit from global memory
        for (size_t k = 0; k < 16; k++) {
          msg[k & 0xf] = word_from_le_bytes(ptr0 + (k << 2));
        }

        // send initial hash state to compressor kernel
        i_pipe0::write(cv0[0]);
        i_pipe0::write(cv0[1]);
        i_pipe0::write(cv0[2]);
        i_pipe0::write(cv0[3]);
        i_pipe0::write(cv0[4]);
        i_pipe0::write(cv0[5]);
        i_pipe0::write(cv0[6]);
        i_pipe0::write(cv0[7]);
        i_pipe0::write(IV[0]);
        i_pipe0::write(IV[1]);
        i_pipe0::write(IV[2]);
        i_pipe0::write(IV[3]);
        i_pipe0::write(static_cast<uint32_t>(chunk_id0 & 0xffffffff));
        i_pipe0::write(static_cast<uint32_t>(chunk_id0 >> 32));
        i_pipe0::write(BLOCK_LEN);
        if (j == 0ul) {
          i_pipe0::write(CHUNK_START);
        } else if (j == 15ul) {
          i_pipe0::write(CHUNK_END);
        } else {
          i_pipe0::write(0);
        }

        // send sixteen message words to compressor kernel
        i_pipe1::write(msg[0]);
        i_pipe1::write(msg[1]);
        i_pipe1::write(msg[2]);
        i_pipe1::write(msg[3]);
        i_pipe1::write(msg[4]);
        i_pipe1::write(msg[5]);
        i_pipe1::write(msg[6]);
        i_pipe1::write(msg[7]);
        i_pipe1::write(msg[8]);
        i_pipe1::write(msg[9]);
        i_pipe1::write(msg[10]);
        i_pipe1::write(msg[11]);
        i_pipe1::write(msg[12]);
        i_pipe1::write(msg[13]);
        i_pipe1::write(msg[14]);
        i_pipe1::write(msg[15]);

        // --- end compressing first chunk's message block ---

        // --- begin compressing second chunk's message block ---

        // #pragma unroll 16 // burst coalesced read of 512 -bit from global
        // memory
        //         for (size_t k = 0; k < 16; k++) {
        //           msg[k & 0xf] = word_from_le_bytes(ptr1 + (k << 2));
        //         }

        //         [[intel::ivdep]] for (size_t k = 0; k < 8; k++)
        //         {
        //           i_pipe0::write(cv1[k & 0x7]);
        //           i_pipe1::write(msg[k & 0x7]);
        //         }

        //         [[intel::ivdep]] for (size_t k = 0; k < 4; k++)
        //         {
        //           i_pipe0::write(IV[k & 0x3]);
        //           i_pipe1::write(msg[(8ul + k) & 0xf]);
        //         }

        //         i_pipe0::write(static_cast<uint32_t>(chunk_id1 & 0xffffffff),
        //         success); i_pipe1::write(msg[12]);

        //         i_pipe0::write(static_cast<uint32_t>(chunk_id1 >> 32),
        //         success); i_pipe1::write(msg[13]);

        //         i_pipe0::write(BLOCK_LEN);
        //         i_pipe1::write(msg[14]);

        //         if (j == 0ul) {
        //           i_pipe0::write(CHUNK_START);
        //         } else if (j == 15ul) {
        //           i_pipe0::write(CHUNK_END);
        //         } else {
        //           i_pipe0::write(0);
        //         }

        //         i_pipe1::write(msg[15]);

        // --- end compressing second chunk's message block ---

        // read output chaining value of first chunk's message block
        cv0[0] = o_pipe0::read();
        cv0[1] = o_pipe0::read();
        cv0[2] = o_pipe0::read();
        cv0[3] = o_pipe0::read();
        cv0[4] = o_pipe0::read();
        cv0[5] = o_pipe0::read();
        cv0[6] = o_pipe0::read();
        cv0[7] = o_pipe0::read();
        // read output chaining value of second chunk's message block
        // [[intel::ivdep]] for (size_t k = 0; k < 8; k++)
        // {
        //   // note, pipe reads are issued blocking
        //   cv1[k & 0x7] = o_pipe0::read();
        // }
      }

      // two consecutive chunks are compressed, so writing their output chaining
      // values on contiguous memory locations, of global memory
#pragma unroll 8 // should be burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        mem_ptr[(mem_idx << 3) + j] = cv0[j & 0x7];
        // mem_ptr[((mem_idx + 1) << 3) + j] = cv1[j & 0x7];
      }

      // move forward global memory pointer by two places,
      // because two chunks just got compressed
      mem_idx += 1;
    }

    //     // these many intermediate levels ( parent chaining values ) of
    //     blake3
    //     // merkle tree to be computed
    //     //
    //     // note, subtraction of 1 comes from the fact that root of merkle
    //     tree
    //     // is not computed in following for loop
    //     [[intel::fpga_register]] const size_t rounds = bin_log(chunk_count) -
    //     1ul;

    //     // sequentially compute all intermediate nodes blake3 binary merkle
    //     tree
    //     // where leaf nodes ( in terms of chunks output chaining values ) are
    //     // provided in global memory
    //     for (size_t r = 0; r < rounds; r++) {
    //       // offset in global memory, for reading four contiguous output
    //       chaining
    //       // values of four nodes, living on same level of blake3 binary
    //       merkle tree
    //       [[intel::fpga_register]] size_t rd_mem_idx = 0;
    //       // offset in global memory, for writing two contiguous output
    //       chaining
    //       // values of two nodes, living on just above of aforementioned four
    //       // nodes of blake3 binary merkle tree
    //       [[intel::fpga_register]] size_t wr_mem_idx = 0;

    //       // in this level of blake3 of binary merkle tree, these many
    //       intermediate
    //       // chaining values to be computed, from double many input chaining
    //       values
    //       [[intel::fpga_register]] const size_t compress_c = chunk_count >>
    //       (r + 1);

    //       // note, in each iteration, I'm compressing two intermediate nodes
    //       // and producing parent node ( read chaining values )
    //       [[intel::ivdep]] for (size_t c = 0; c < compress_c; c += 1)
    //       {
    //         // --- begin compressing first (two) intermediate node ---

    // #pragma unroll 16 // burst coalesced 512 -bit read from global memory
    //         for (size_t j = 0; j < 16; j++) {
    //           msg[j & 0xf] = mem_ptr[(rd_mem_idx << 3) + j];
    //         }

    //         // send initial hash state to compressor kernel
    //         i_pipe0::write(IV[0]);
    //         i_pipe0::write(IV[1]);
    //         i_pipe0::write(IV[2]);
    //         i_pipe0::write(IV[3]);
    //         i_pipe0::write(IV[4]);
    //         i_pipe0::write(IV[5]);
    //         i_pipe0::write(IV[6]);
    //         i_pipe0::write(IV[7]);
    //         i_pipe0::write(IV[0]);
    //         i_pipe0::write(IV[1]);
    //         i_pipe0::write(IV[2]);
    //         i_pipe0::write(IV[3]);
    //         i_pipe0::write(0u);
    //         i_pipe0::write(0u);
    //         i_pipe0::write(BLOCK_LEN);
    //         i_pipe0::write(PARENT);

    //         // send sixteen message words to compressor kernel
    //         i_pipe1::write(msg[0]);
    //         i_pipe1::write(msg[1]);
    //         i_pipe1::write(msg[2]);
    //         i_pipe1::write(msg[3]);
    //         i_pipe1::write(msg[4]);
    //         i_pipe1::write(msg[5]);
    //         i_pipe1::write(msg[6]);
    //         i_pipe1::write(msg[7]);
    //         i_pipe1::write(msg[8]);
    //         i_pipe1::write(msg[9]);
    //         i_pipe1::write(msg[10]);
    //         i_pipe1::write(msg[11]);
    //         i_pipe1::write(msg[12]);
    //         i_pipe1::write(msg[13]);
    //         i_pipe1::write(msg[14]);
    //         i_pipe1::write(msg[15]);

    //         rd_mem_idx += 2;

    //         // --- end compressing first (two) intermediate node ---

    //         // --- begin compressing next (two) intermediate node ---

    //         // #pragma unroll 16 // burst coalesced 512 -bit read from global
    //         memory
    //         //         for (size_t j = 0; j < 16; j++) {
    //         //           msg[j & 0xf] = mem_ptr[(rd_mem_idx << 3) + j];
    //         //         }

    //         //         [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
    //         //         {
    //         //           i_pipe0::write(IV[j & 0x7]);
    //         //           i_pipe1::write(msg[j & 0x7]);
    //         //         }

    //         //         [[intel::ivdep]] for (size_t j = 0; j < 4; j++)
    //         //         {
    //         //           i_pipe0::write(IV[j & 0x3]);
    //         //           i_pipe1::write(msg[(8ul + j) & 0xf]);
    //         //         }

    //         //         i_pipe0::write(0u);
    //         //         i_pipe1::write(msg[12]);

    //         //         i_pipe0::write(0u);
    //         //         i_pipe1::write(msg[13]);

    //         //         i_pipe0::write(BLOCK_LEN);
    //         //         i_pipe1::write(msg[14]);

    //         //         i_pipe0::write(PARENT);
    //         //         i_pipe1::write(msg[15]);

    //         //         rd_mem_idx += 2;

    //         // --- end compressing next (two) intermediate node ---

    //         // read output chaining values ( of first pair of compressed
    //         nodes )
    //         // from fifo pipe
    //         cv0[0] = o_pipe0::read();
    //         cv0[1] = o_pipe0::read();
    //         cv0[2] = o_pipe0::read();
    //         cv0[3] = o_pipe0::read();
    //         cv0[4] = o_pipe0::read();
    //         cv0[5] = o_pipe0::read();
    //         cv0[6] = o_pipe0::read();
    //         cv0[7] = o_pipe0::read();

    //         // then write those back to global memory
    // #pragma unroll 8 // 256 -bit wide burst coalesced access
    //         for (size_t j = 0; j < 8; j++) {
    //           mem_ptr[(wr_mem_idx << 3) + j] = cv0[j & 0x7];
    //         }

    //         // read output chaining values ( of second pair of compressed
    //         nodes )
    //         // from fifo pipe
    //         //         [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
    //         //         {
    //         //           // note, pipe reads are issued blocking
    //         //           cv1[j & 0x7] = o_pipe0::read();
    //         //         }

    //         //         // then write those back to global memory
    //         // #pragma unroll 8 // 256 -bit wide burst coalesced write
    //         //         for (size_t j = 0; j < 8; j++) {
    //         //           mem_ptr[((wr_mem_idx + 1) << 3) + j] = cv1[j & 0x7];
    //         //         }

    //         wr_mem_idx += 1;
    //       }
    //     }

    //     // --- begin computing root ( i.e. digest ) of blake3 binary merkle
    //     tree ---

    // #pragma unroll 16 // 512 -bit burst coalesced read from global memory
    //     for (size_t j = 0; j < 16; j++) {
    //       msg[j & 0xf] = mem_ptr[j];
    //     }

    //     // send initial hash state to compressor kernel
    //     i_pipe0::write(IV[0]);
    //     i_pipe0::write(IV[1]);
    //     i_pipe0::write(IV[2]);
    //     i_pipe0::write(IV[3]);
    //     i_pipe0::write(IV[4]);
    //     i_pipe0::write(IV[5]);
    //     i_pipe0::write(IV[6]);
    //     i_pipe0::write(IV[7]);
    //     i_pipe0::write(IV[0]);
    //     i_pipe0::write(IV[1]);
    //     i_pipe0::write(IV[2]);
    //     i_pipe0::write(IV[3]);
    //     i_pipe0::write(0u);
    //     i_pipe0::write(0u);
    //     i_pipe0::write(BLOCK_LEN);
    //     i_pipe0::write(PARENT | ROOT);

    //     // send sixteen message words to compressor kernel
    //     i_pipe1::write(msg[0]);
    //     i_pipe1::write(msg[1]);
    //     i_pipe1::write(msg[2]);
    //     i_pipe1::write(msg[3]);
    //     i_pipe1::write(msg[4]);
    //     i_pipe1::write(msg[5]);
    //     i_pipe1::write(msg[6]);
    //     i_pipe1::write(msg[7]);
    //     i_pipe1::write(msg[8]);
    //     i_pipe1::write(msg[9]);
    //     i_pipe1::write(msg[10]);
    //     i_pipe1::write(msg[11]);
    //     i_pipe1::write(msg[12]);
    //     i_pipe1::write(msg[13]);
    //     i_pipe1::write(msg[14]);
    //     i_pipe1::write(msg[15]);

    //     // output chaining value of root node i.e. blake3 digest
    //     // being read from compressor kernel
    //     //
    //     // blocking call !
    //     cv0[0] = o_pipe0::read();
    //     cv0[1] = o_pipe0::read();
    //     cv0[2] = o_pipe0::read();
    //     cv0[3] = o_pipe0::read();
    //     cv0[4] = o_pipe0::read();
    //     cv0[5] = o_pipe0::read();
    //     cv0[6] = o_pipe0::read();
    //     cv0[7] = o_pipe0::read();

    //     // --- end computing root ( i.e. digest ) of blake3 binary merkle
    //     tree ---

    //     // write digest back ( in little endian byte form ) to allocated 32
    //     -bytes (
    //     // global ) memory
    //     words_to_le_bytes(sycl::private_ptr<uint32_t>(cv0), o_ptr);
  });

  sycl::event evt1 = q.single_task<kernelBlake3Compressor>([=
  ]() [[intel::kernel_args_restrict]] {
    // orchestrator kernel needs to perform `rounds` -many compressions
    [[intel::fpga_register]] const size_t rounds =
      (chunk_count << 4) /* + (chunk_count - 1ul) */;

    [[intel::fpga_memory("BLOCK_RAM"),
      intel::numbanks(16),
      intel::bankwidth(4)]] sycl::uint4 state[4];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::numbanks(16),
      intel::bankwidth(4)]] uint32_t msg[16];

    for (size_t i = 0; i < rounds; i++) {
      // initial hash state ( 64 -bytes ) from orchestrator kernel
      uint32_t a = i_pipe0::read();
      uint32_t b = i_pipe0::read();
      uint32_t c = i_pipe0::read();
      uint32_t d = i_pipe0::read();
      // first row of hash state as 128 -bit vector
      state[0] = sycl::uint4(a, b, c, d);

      a = i_pipe0::read();
      b = i_pipe0::read();
      c = i_pipe0::read();
      d = i_pipe0::read();
      // second row of hash state as 128 -bit vector
      state[1] = sycl::uint4(a, b, c, d);

      a = i_pipe0::read();
      b = i_pipe0::read();
      c = i_pipe0::read();
      d = i_pipe0::read();
      // third row of hash state as 128 -bit vector
      state[2] = sycl::uint4(a, b, c, d);

      a = i_pipe0::read();
      b = i_pipe0::read();
      c = i_pipe0::read();
      d = i_pipe0::read();
      // fourth row of hash state as 128 -bit vector
      state[3] = sycl::uint4(a, b, c, d);

      // input message words ( 64 -bytes ) from orchestrator kernel
      msg[0] = i_pipe1::read();
      msg[1] = i_pipe1::read();
      msg[2] = i_pipe1::read();
      msg[3] = i_pipe1::read();
      msg[4] = i_pipe1::read();
      msg[5] = i_pipe1::read();
      msg[6] = i_pipe1::read();
      msg[7] = i_pipe1::read();
      msg[8] = i_pipe1::read();
      msg[9] = i_pipe1::read();
      msg[10] = i_pipe1::read();
      msg[11] = i_pipe1::read();
      msg[12] = i_pipe1::read();
      msg[13] = i_pipe1::read();
      msg[14] = i_pipe1::read();
      msg[15] = i_pipe1::read();

      // compress sixteen message words into hash state, produced output
      // chaining value lives on first 8 words of hash state ( i.e. on first two
      // 128 -bit vectors )
      compress(state, msg);

      // finally send 8 word output chaining value, as result of compression,
      // to orchestrator kernel
      o_pipe0::write(state[0].x());
      o_pipe0::write(state[0].y());
      o_pipe0::write(state[0].z());
      o_pipe0::write(state[0].w());
      o_pipe0::write(state[1].x());
      o_pipe0::write(state[1].y());
      o_pipe0::write(state[1].z());
      o_pipe0::write(state[1].w());
    }
  });

  q.ext_oneapi_submit_barrier({ evt0, evt1 }).wait();
  sycl::free(mem, q);

  // calculate kernel execution time ( in nanoseconds ), as mean execution
  // time of two concurrently execution kernels, which communicate over message
  // passing pipe interface
  if (ts != nullptr) {
    sycl::cl_ulong ts_ = 0;

    ts_ += time_event(evt0);
    ts_ += time_event(evt1);

    *ts = ts_ / 2;
  }
}
}
