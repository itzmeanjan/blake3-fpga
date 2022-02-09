#pragma once
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

// Pipe to be used for sending initial hash state of 64 -bytes to compressor
// kernel from orchestrator kernel
using i_pipe0 = sycl::ext::intel::pipe<class HashStatePipe, uint32_t, 16>;
// Pipe to be used for sending ( total ) 64 -bytes message words to compressor
// kernel from orchestrator kernel
using i_pipe1 = sycl::ext::intel::pipe<class MessageWordsPipe, uint32_t, 16>;
// Pipe to be used for sending 32 -bytes output chaining value as result of
// compression, from compressor to orchestrator kernel
using o_pipe0 = sycl::ext::intel::pipe<class ChainingValuePipe, uint32_t, 8>;

// Compile time check for template function argument; checks whether
// requested rotation bit positions for message word ( = 32 -bit )
// is in range [0, 32)
static constexpr bool
valid_bit_pos(size_t bit_pos)
{
  return bit_pos >= 0 && bit_pos < 32;
}

// Rotates ( read circular right shift ) 32 -bit wide BLAKE3 message word
// rightwards, by N -bit places ( note 0 <= N < 32 ), where N must
// be compile time constant !
template<size_t bit_pos>
static inline const uint32_t
rotr(uint32_t word) requires(valid_bit_pos(bit_pos))
{
  return (word >> bit_pos) | (word << (32 - bit_pos));
}

// Mixes two message words into 64 -bytes wide state either column-wise/
// diagonally
//
// See
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L42-L52
static inline void
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

// BLAKE3 round, applied 7 times for mixing sixteen message words ( = total 64
// -bytes ) into hash state, both column-wise and diagonally !
//
// See
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L54-L65
static inline void
rnd(sycl::private_ptr<uint32_t> state, sycl::private_ptr<uint32_t> msg)
{
  // Mixing first eight message words of block into state column-wise
  g(state, 0, 4, 8, 12, msg[0], msg[1]);
  g(state, 1, 5, 9, 13, msg[2], msg[3]);
  g(state, 2, 6, 10, 14, msg[4], msg[5]);
  g(state, 3, 7, 11, 15, msg[6], msg[7]);

  // Mixing last eight message words of block into state diagonally
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
permute(sycl::private_ptr<uint32_t> msg_words)
{
  // additional array memory ( = 64 -bytes ) for helping in message word
  // permutation
  //
  // probably going to be optimized such that it's not synthesized !
  [[intel::fpga_memory("BLOCK_RAM")]] uint32_t permuted[16];

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = msg_words[MSG_PERMUTATION[i]];
  }

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    msg_words[i] = permuted[i];
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
compress(sycl::private_ptr<uint32_t> state,    // hash state
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
#pragma unroll 8 // fully parallelize loop, not data dependency
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[i + 8];
    // notice I'm skipping
    // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
    // because that statement doesn't dictate what output chaining value of this
    // message block will be !
    //
    // So it's safe to do so !
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
words_to_le_bytes(const sycl::device_ptr<uint32_t> msg_words,
                  sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    word_to_le_bytes(msg_words[i], output + (i << 2));
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

// Computes how many trailing zero bits are present in n ( > 0 )
static inline const size_t
ctz(size_t n)
{
  size_t cnt = 0ul;

  while ((n & 0b1ul) == 0ul) {
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

  // allocated on device memory, so that all those intermediate ( output )
  // chaining values can be bufferred !
  //
  // Note, each output chaining value is 32 -bytes
  //
  // During BLAKE3 hash computation with `N` -many 1024 -bytes message
  // chunks, at a time, maximum log2(N) -many output chaining values will
  // be required to be bufferred !
  const size_t mem_size = (bin_log(chunk_count) << 3) * sizeof(uint32_t);
  uint32_t* mem = static_cast<uint32_t*>(sycl::malloc_device(mem_size, q));

  sycl::event evt0 = q.single_task<kernelBlake3Orchestrator>([=
  ]() [[intel::kernel_args_restrict]] {
    // Following best practices guide, so that it hints compiler that following
    // three allocations are on device memory & LSU doesn't need to interface
    // with host memory unnecessarily
    sycl::device_ptr<sycl::uchar> i_ptr = sycl::device_ptr<sycl::uchar>(input);
    sycl::device_ptr<uint32_t> mem_ptr = sycl::device_ptr<uint32_t>(mem);
    sycl::device_ptr<sycl::uchar> o_ptr = sycl::device_ptr<sycl::uchar>(digest);

    // mem_idx ∈ [0, log2(N)] | N = chunk_count
    [[intel::fpga_register]] size_t mem_idx = 0ul;
    // temporary storage for output input/ output chaining values
    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t cv[8];
    // temporary storage for input message words ( = 64 -bytes ),
    // so that burst coalesced 512 -bit access can be performed from memory
    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg[16];

    for (size_t i = 0; i < chunk_count; i++) {
      // for each chunk, first message block ( = 64 -bytes )
      // has ( same ) constant initial hash values for first 8 words
#pragma unroll 8
      for (size_t j = 0; j < 8; j++) {
        cv[j] = IV[j];
      }

      // each chunk ( = 1024 -bytes ), has 16 message blocks, each of 64 -bytes
      // width, which are required to be processed sequentially
      //
      // To be more specific, j-th chunk's input chaining values ( = first 32
      // -bytes ) are taken from (j-1) -th chunk's output chaining values ( 32
      // -bytes ) | j = {1, 2, 3, ... 15}
      //
      // Note, for j = 0, input chaining values are already initialised in
      // above unrolled loop !
      for (size_t j = 0; j < 16; j++) {
        sycl::device_ptr<sycl::uchar> ptr_ = i_ptr + (i << 10) + (j << 6);

        // in following statements, 64 -bytes input hash state & 64 -bytes input
        // message block is passed via pipe ( to kernel which compresses message
        // blocks and produces output chaining value of 32 -bytes ) !

#pragma unroll 16
        for (size_t k = 0; k < 16; k++) {
          msg[k] = word_from_le_bytes(ptr_ + (k << 2));
        }

        for (size_t k = 0; k < 8; k++) {
          i_pipe0::write(cv[k]);
          i_pipe1::write(msg[k]);
        }

        for (size_t k = 0; k < 4; k++) {
          i_pipe0::write(IV[k]);
          i_pipe1::write(msg[8ul + k]);
        }

        i_pipe0::write(static_cast<uint32_t>(i & 0xffffffff));
        i_pipe1::write(msg[12]);

        i_pipe0::write(static_cast<uint32_t>(i >> 32));
        i_pipe1::write(msg[13]);

        i_pipe0::write(BLOCK_LEN);
        i_pipe1::write(msg[14]);

        // ensure that proper message block denoter flags are passed
        // before compression
        if (j == 0ul) {
          i_pipe0::write(CHUNK_START);
        } else if (j == 15ul) {
          i_pipe0::write(CHUNK_END);
        } else {
          i_pipe0::write(0);
        }

        i_pipe1::write(msg[15]);

        // in following statements, 32 -bytes output chaining value of
        // compressed message block is being read !

        for (size_t k = 0; k < 8; k++) {
          cv[k] = o_pipe0::read();
        }
      }

      [[intel::fpga_register]] size_t zeros = ctz(i + 1ul);

      while (zeros > 0) {
        mem_idx--;

        for (size_t j = 0; j < 8; j++) {
          i_pipe0::write(IV[j]);
          i_pipe1::write(mem_ptr[(mem_idx << 3) + j]);
        }

        for (size_t k = 0; k < 4; k++) {
          i_pipe0::write(IV[k]);
          i_pipe1::write(cv[k]);
        }

        i_pipe0::write(0u);
        i_pipe1::write(cv[4]);

        i_pipe0::write(0u);
        i_pipe1::write(cv[5]);

        i_pipe0::write(BLOCK_LEN);
        i_pipe1::write(cv[6]);

        if (i + 1ul == chunk_count && zeros == 1) {
          i_pipe0::write(PARENT | ROOT);
        } else {
          i_pipe0::write(PARENT);
        }

        i_pipe1::write(cv[7]);

        for (size_t j = 0; j < 8; j++) {
          cv[j] = o_pipe0::read();
        }

        zeros--;
      }

#pragma unroll 8
      for (size_t j = 0; j < 8; j++) {
        mem_ptr[(mem_idx << 3) + j] = cv[j];
      }

      mem_idx++;
    }

    words_to_le_bytes(mem_ptr, o_ptr);
  });

  sycl::event evt1 = q.single_task<kernelBlake3Compressor>([=
  ]() [[intel::kernel_args_restrict]] {
    // orchestrator kernel needs to perform `rounds` -many compressions
    [[intel::fpga_register]] const size_t rounds =
      (chunk_count << 4) + (chunk_count - 1ul);

    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t state[16];
    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg[16];

    for (size_t i = 0; i < rounds; i++) {
      // get initial hash state ( 64 -bytes ) and input message words
      // ( 64 -bytes ) from orchestrator kernel
      for (size_t j = 0; j < 16; j++) {
        state[j] = i_pipe0::read();
        msg[j] = i_pipe1::read();
      }

      // compress sixteen message words into hash state, output chaining value
      // lives on first 8 words of hash state
      compress(state, msg);

      // finally send 8 word output chaining value, as result of compression,
      // to orchestrator kernel
      for (size_t j = 0; j < 8; j++) {
        o_pipe0::write(state[j]);
      }
    }
  });

  q.ext_oneapi_submit_barrier({ evt0, evt1 }).wait();
  sycl::free(mem, q);

  if (ts != nullptr) {
  }
}
}
