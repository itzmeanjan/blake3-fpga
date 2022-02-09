#pragma once
#include "common.hpp"
#include <cassert>

namespace blake3 {

// just to avoid kernel name mangling issue in optimization report
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
// BLAKE3 is 32 -bit ) by 7 mixing rounds and 6 permutation rounds
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1641-L1703
static inline void
compress(sycl::private_ptr<uint32_t> state,     // hash state
         sycl::private_ptr<uint32_t> msg_words, // input message
         const uint64_t counter,
         const uint32_t block_len,
         const uint32_t flags)
{
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    state[8 + i] = IV[i];
  }

  state[12] = static_cast<uint32_t>(counter & 0xffffffff);
  state[13] = static_cast<uint32_t>(counter >> 32);
  state[14] = block_len;
  state[15] = flags;
  // --- initialising hash state, ends ---

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
  // and keeping it in first 8 message words of hash state
#pragma unroll 8 // fully parallelize loop, not data dependency
  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[i + 8];
    // notice I'm skipping
    // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
    // because that statement doesn't dictate what output chaining value of this
    // block will be !
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
words_to_le_bytes(const sycl::private_ptr<uint32_t> msg_words,
                  sycl::device_ptr<sycl::uchar> output)
{
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    word_to_le_bytes(msg_words[i], output + (i << 2));
  }
}

// Sequentially compresses all sixteen message blocks present in a 1024
// -bytes wide BLAKE3 chunk and produces 32 -bytes output chaining value
// of this chunk, which will be used for computing parent nodes of BLAKE3
// merkle tree
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1790-L1842
static inline void
chunkify(
  const uint64_t chunk_counter,
  sycl::private_ptr<uint32_t> state,     // hash state
  sycl::private_ptr<uint32_t> msg_blocks // 256 message words ( i.e. 16 message
                                         // blocks ) to be compressed
)
{
  // initialise hash state for first message block of chunk
  //
  // for all further message block compression, previous block's
  // output chaining values to be used as input chaining values
  //
  // that means, output chaining value obtained after any compress function
  // invocation, should be living over first 8 words of hash state
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    state[i] = IV[i];
  }

  // compress first message block of 1024 -bytes wide chunk
  compress(state, msg_blocks, chunk_counter, BLOCK_LEN, CHUNK_START);

  // intermediate 14 message blocks to be compressed in order
  for (size_t j = 1; j < 15; j++) {
    compress(state, msg_blocks + (j << 4), chunk_counter, BLOCK_LEN, 0);
  }

  // compress last message block of 1024 -bytes wide chunk
  compress(state, msg_blocks + (15u << 4), chunk_counter, BLOCK_LEN, CHUNK_END);
  //
  // after last message block of chunk is compressed, output chaining value of
  // whole chunk can be found on first 8 words of hash state
}

// Computes chaining value for some parent ( intermediate, but non-root ) node
// of BLAKE3 merkle tree, by compressing sixteen input message words
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1844-L1865
static inline void
parent_cv(sycl::private_ptr<uint32_t> state, // hash state
          sycl::private_ptr<uint32_t> msg_words,
          const sycl::device_ptr<uint32_t> left_cv,
          const sycl::device_ptr<uint32_t> right_cv,
          const uint32_t flags)
{
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    // setting first 8 words of hash state
    state[i] = IV[i];
  }

#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    // setting total of 16 message words of
    // input message to be consumed into hash state
    // by following invocation of compress function
    msg_words[i] = left_cv[i];
    msg_words[i + 8] = right_cv[i];
  }

  compress(state, msg_words, 0, BLOCK_LEN, flags | PARENT);
  //
  // find output chaining value of this parent, in first 8 words
  // of hash state
}

// Computes BLAKE3 merkle tree's root chaining value ( 32 -bytes ) by
// compressing two immediate children node's chaining values ( each chaining
// value 32 -bytes )
//
// Note this chaining value is nothing but BLAKE3 digest
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1867-L1874
static inline void
root_cv(sycl::private_ptr<uint32_t> state, // hash state
        sycl::private_ptr<uint32_t> msg_words,
        const sycl::device_ptr<uint32_t> left_cv,
        const sycl::device_ptr<uint32_t> right_cv)
{
  parent_cv(state, msg_words, left_cv, right_cv, ROOT);
}

// Binary logarithm of n, when n = 2 ^ i | i = {1, 2, ...}
static inline const size_t
bin_log(size_t n)
{
  [[intel::fpga_register]] size_t cnt = 0;

  while (n > 1) {
    n >>= 1;
    cnt++;
  }

  return cnt;
}

// BLAKE3 hash function, can be used when chunk count is power of 2
//
// Note, chunk count is preferred to be relatively large number ( say >= 2^20 )
// because this function is supposed to be executed on accelerator
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

  const size_t mem_size = static_cast<size_t>(BLOCK_LEN) * chunk_count;
  uint32_t* mem = static_cast<uint32_t*>(sycl::malloc_device(mem_size, q));

  sycl::event evt = q.single_task<kernelBlake3Hash>([=
  ]() [[intel::kernel_args_restrict]] {
    // explicitly let compiler know that all (runtime allocated) memory accessed
    // from kernel are living on accelerator, so LSU doesn't need to interface
    // with host memory at all !
    sycl::device_ptr<sycl::uchar> i_ptr = sycl::device_ptr<sycl::uchar>(input);
    sycl::device_ptr<uint32_t> mem_ptr = sycl::device_ptr<uint32_t>(mem);
    sycl::device_ptr<sycl::uchar> o_ptr = sycl::device_ptr<sycl::uchar>(digest);

    {
      [[intel::fpga_register]] const size_t mem_offset = chunk_count << 3;

      [[intel::fpga_memory("BLOCK_RAM")]] uint32_t state_0[16];
      [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg_blocks_0[256];

      // compress all chunks so that each chunk produces a single output
      // chaining value of 32 -bytes, which are to be later used for
      // computing parent chaining values ( i.e. Binary Merklization, see 👇 )
      [[intel::ivdep]] for (size_t i = 0; i < chunk_count; i++)
      {
#pragma unroll 16
        for (size_t j = 0; j < 256; j++) {
          msg_blocks_0[j] = word_from_le_bytes(i_ptr + (i << 10) + (j << 2));
        }

        chunkify(static_cast<uint64_t>(i),
                 sycl::private_ptr<uint32_t>(state_0),
                 sycl::private_ptr<uint32_t>(msg_blocks_0));

#pragma unroll 8
        for (size_t j = 0; j < 8; j++) {
          mem_ptr[mem_offset + (i << 3) + j] = state_0[j];
        }
      }
    }

    {
      // `chunk_count` -many leaf nodes of BLAKE3's internal binary merkle
      // tree to be merged in `rounds` -many dispatch rounds such that we
      // reach root node ( i.e. chaining value ) level of merkle tree
      [[intel::fpga_register]] const size_t rounds = bin_log(chunk_count) - 1ul;

      [[intel::fpga_register]] size_t parent_count;
      [[intel::fpga_register]] size_t read_offset;
      [[intel::fpga_register]] size_t write_offset;

      [[intel::fpga_memory("BLOCK_RAM")]] uint32_t state_2[16];
      [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg_words_0[16];

      // process each level of nodes of BLAKE3's internal binary merkle tree
      // in order
      //
      // as round 0 must complete before round 1 can begin, I can't let
      // compiler coalesce following nested loop construction
      for (size_t r = 0; r < rounds; r++) {
        parent_count = chunk_count >> (r + 1);
        read_offset = (chunk_count << 3) >> r;
        write_offset = read_offset >> 1;

        // computing output chaining values of all children nodes of
        // some level of BlAKE3 binary merkle tree
        // can occur in parallel, no data dependency !
        //
        // but ensure that following loop completes execution for (r = 0, see
        // above) before r = 1's body execution can begin, due to presence of
        // critical data dependency !
        [[intel::ivdep]] for (size_t i = 0; i < parent_count; i += 1)
        {
          parent_cv(sycl::private_ptr<uint32_t>(state_2),
                    sycl::private_ptr<uint32_t>(msg_words_0),
                    mem_ptr + read_offset + ((i << 1) << 3),
                    mem_ptr + read_offset + (((i << 1) + 1) << 3),
                    0);

#pragma unroll 8
          for (size_t j = 0; j < 8; j++) {
            mem_ptr[write_offset + (i << 3) + j] = state_2[j];
          }
        }
      }
    }

    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t state_4[16];
    [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg_words_2[16];

    // finally compute root chaining value ( which is digest ) of BLAKE3
    // merkle tree
    root_cv(sycl::private_ptr<uint32_t>(state_4),
            sycl::private_ptr<uint32_t>(msg_words_2),
            mem_ptr + 16ul + (0ul << 3),
            mem_ptr + 16ul + (1ul << 3));

    // write 32 -bytes BLAKE3 digest back to allocated memory !
    words_to_le_bytes(sycl::private_ptr<uint32_t>(state_4), o_ptr);
  });

  evt.wait();
  sycl::free(mem, q);

  // if ts is non-null ensure that SYCL queue has profiling
  // enabled, otherwise following lines should panic !
  if (ts != nullptr) {
    // write back total kernel execution time
    *ts = time_event(evt);
  }
}
}
