#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
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
constexpr sycl::uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constexpr size_t CHUNK_LEN = 1024;   // bytes
constexpr size_t OUT_LEN = 32;       // bytes
constexpr sycl::uint BLOCK_LEN = 64; // bytes

// Mixing rounds
constexpr size_t ROUNDS = 7;

// BLAKE3 flags
constexpr sycl::uint CHUNK_START = 1 << 0;
constexpr sycl::uint CHUNK_END = 1 << 1;
constexpr sycl::uint PARENT = 1 << 2;
constexpr sycl::uint ROOT = 1 << 3;

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
static inline const sycl::uint
rotr(sycl::uint word) requires(valid_bit_pos(bit_pos))
{
  return (word >> bit_pos) | (word << (32 - bit_pos));
}

// BLAKE3 round, applied 7 times for mixing sixteen message words ( = total 64
// -bytes ) into hash state, both column-wise and diagonally !
//
// See
// https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L54-L65
static inline void
rnd(sycl::uint* const __restrict state, const sycl::uint* const __restrict msg)
{
// Mixing first eight message words of block into state column-wise
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    state[0 + i] += state[4 + i] + msg[i << 1];
    state[12 + i] = rotr<16>(state[12 + i] ^ state[0 + i]);
    state[8 + i] += state[12 + i];
    state[4 + i] = rotr<12>(state[4 + i] ^ state[8 + i]);

    state[0 + i] += state[4 + i] + msg[(i << 1) + 1];
    state[12 + i] = rotr<8>(state[12 + i] ^ state[0 + i]);
    state[8 + i] += state[12 + i];
    state[4 + i] = rotr<7>(state[4 + i] ^ state[8 + i]);
  }

  // following three code blocks help in diagonalizing 4x4 hash state matrix
  //
  // note, row 0 doesn't need to be touched !
  //
  // diagonalize row 1 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp = state[4];
    for (size_t i = 4; i < 7; i++) {
      state[i] = state[i + 1];
    }
    state[7] = tmp;
  }

  // diagonalize row 2 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp0 = state[8];
    [[intel::fpga_register]] const sycl::uint tmp1 = state[9];
    for (size_t i = 8; i < 10; i++) {
      state[i] = state[i + 2];
    }
    state[10] = tmp0;
    state[11] = tmp1;
  }

  // diagonalize row 3 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp = state[15];
    for (size_t i = 15; i > 12; i--) {
      state[i] = state[i - 1];
    }
    state[12] = tmp;
  }

// Mixing last eight message words of block into state diagonally
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    state[0 + i] += state[4 + i] + msg[8 + (i << 1)];
    state[12 + i] = rotr<16>(state[12 + i] ^ state[0 + i]);
    state[8 + i] += state[12 + i];
    state[4 + i] = rotr<12>(state[4 + i] ^ state[8 + i]);

    state[0 + i] += state[4 + i] + msg[8 + (i << 1) + 1];
    state[12 + i] = rotr<8>(state[12 + i] ^ state[0 + i]);
    state[8 + i] += state[12 + i];
    state[4 + i] = rotr<7>(state[4 + i] ^ state[8 + i]);
  }

  // following three code blocks help in un-diagonalizing 4x4 hash state matrix
  //
  // note, row 0 doesn't need to be touched !
  //
  // un-diagonalize row 1 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp = state[7];
    for (size_t i = 7; i > 4; i--) {
      state[i] = state[i - 1];
    }
    state[4] = tmp;
  }

  // un-diagonalize row 2 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp0 = state[8];
    [[intel::fpga_register]] const sycl::uint tmp1 = state[9];
    for (size_t i = 8; i < 10; i++) {
      state[i] = state[i + 2];
    }
    state[10] = tmp0;
    state[11] = tmp1;
  }

  // un-diagonalize row 3 of 4x4 state matrix
  {
    [[intel::fpga_register]] const sycl::uint tmp = state[12];
    for (size_t i = 12; i < 15; i++) {
      state[i] = state[i + 1];
    }
    state[15] = tmp;
  }
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
permute(sycl::uint* const msg_words)
{
  // additional array memory ( = 64 -bytes ) for helping in message word
  // permutation
  //
  // probably going to be optimized such that it's not synthesized !
  [[intel::fpga_memory]] sycl::uint permuted[16];

#pragma unroll 8
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = msg_words[MSG_PERMUTATION[i]];
  }

#pragma unroll 8
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
compress(const sycl::uint* in_cv,
         sycl::uint* const __restrict msg_words,
         const sycl::ulong counter,
         const sycl::uint block_len,
         const sycl::uint flags,
         sycl::uint* const __restrict out_cv)
{
  // initial hash state, which will consume all sixteen message words ( = 64
  // -bytes total ) and produce output chaining value of this message block
  [[intel::fpga_memory("MLAB")]] sycl::uint state[16];

  // --- initialising hash state, begins ---
#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    state[i] = in_cv[i];
  }

#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    state[8 + i] = IV[i];
  }

  state[12] = static_cast<sycl::uint>(counter & 0xffffffff);
  state[13] = static_cast<sycl::uint>(counter >> 32);
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

#pragma unroll 8 // fully parallelize loop, not data dependency
  for (size_t i = 0; i < 8; i++) {
    state[i] = state[i] ^ state[i + 8];
    // notice I'm skipping
    // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs#L118
    // because that statement doesn't dictate what output chaining value of this
    // block will be !
    //
    // So it's safe to do so !
  }

  // output chaining value of this block to be used as
  // input chaining value for next block in same chunk
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    out_cv[i] = state[i];
    out_cv[i + 4] = state[i + 4];
  }
}

// Four consecutive little endian bytes are interpreted as 32 -bit unsigned
// integer i.e. BLAKE3 message word
static inline const sycl::uint
word_from_le_bytes(const sycl::uchar* const input)
{
  return static_cast<sycl::uint>(input[3]) << 24 |
         static_cast<sycl::uint>(input[2]) << 16 |
         static_cast<sycl::uint>(input[1]) << 8 |
         static_cast<sycl::uint>(input[0]) << 0;
}

// 64 little endian input bytes of a message block to be interpreted as sixteen
// BLAKE3 words ( = uint32_t each )
static inline void
words_from_le_bytes(const sycl::uchar* const __restrict input,
                    sycl::uint* const __restrict msg_words)
{
#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    msg_words[i] = word_from_le_bytes(input + (i << 2));
  }
}

// One 32 -bit BLAKE3 word is converted to four consecutive little endian bytes
static inline void
word_to_le_bytes(const sycl::uint word, sycl::uchar* const output)
{
#pragma unroll 4
  for (size_t i = 0; i < 4; i++) {
    output[i] = static_cast<sycl::uchar>((word >> (i << 3)) & 0xff);
  }
}

// Eight consecutive BLAKE3 message words are converted to 32 little endian
// bytes
static inline void
words_to_le_bytes(const sycl::uint* const __restrict msg_words,
                  sycl::uchar* const __restrict output)
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
void
chunkify(const sycl::uint* const __restrict key_words,
         const sycl::ulong chunk_counter,
         const sycl::uint flags,
         const sycl::uchar* const __restrict input,
         sycl::uint* const __restrict out_cv)
{
  [[intel::fpga_memory("BLOCK_RAM")]] sycl::uint in_cv[8];
  [[intel::fpga_memory("BLOCK_RAM")]] sycl::uint priv_out_cv[8];
  [[intel::fpga_memory("BLOCK_RAM")]] sycl::uint msg_words[16];

#pragma unroll 8 // attempt to fully parallelize array initialization !
  for (size_t i = 0; i < 8; i++) {
    in_cv[i] = key_words[i];
  }

  // --- begin processing first message block ---
  words_from_le_bytes(input, msg_words);
  compress(in_cv,
           msg_words,
           chunk_counter,
           BLOCK_LEN,
           flags | CHUNK_START,
           priv_out_cv);

#pragma unroll 8 // copying between array can be fully parallelized !
  for (size_t j = 0; j < 8; j++) {
    in_cv[j] = priv_out_cv[j];
  }
  // --- end processing first message block ---

  // process intermediate ( read non-boundary ) 14 message blocks
  for (size_t i = 1; i < 15; i++) {
    words_from_le_bytes(input + i * BLOCK_LEN, msg_words);
    compress(in_cv, msg_words, chunk_counter, BLOCK_LEN, flags, priv_out_cv);

#pragma unroll 8 // copying between array can be fully parallelized !
    for (size_t j = 0; j < 8; j++) {
      in_cv[j] = priv_out_cv[j];
    }
  }

  // --- begin processing last message block ---
  words_from_le_bytes(input + 15 * BLOCK_LEN, msg_words);
  compress(
    in_cv, msg_words, chunk_counter, BLOCK_LEN, flags | CHUNK_END, out_cv);
  // --- end processing last message block ---
}

// Computes chaining value for some parent ( intermediate, but non-root ) node
// of BLAKE3 merkle tree, by compressing sixteen input message words
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1844-L1865
static inline void
parent_cv(const sycl::uint* const __restrict left_cv,
          const sycl::uint* const __restrict right_cv,
          const sycl::uint* const __restrict key_words,
          const sycl::uint flags,
          sycl::uint* const __restrict out_cv)
{
  [[intel::fpga_memory("BLOCK_RAM")]] sycl::uint block_words[16];

#pragma unroll 8
  for (size_t i = 0; i < 8; i++) {
    block_words[i] = left_cv[i];
    block_words[i + 8] = right_cv[i];
  }

  compress(key_words, block_words, 0, BLOCK_LEN, flags | PARENT, out_cv);
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
root_cv(const sycl::uint* const __restrict left_cv,
        const sycl::uint* const __restrict right_cv,
        const sycl::uint* const __restrict key_words,
        sycl::uint* const __restrict out_cv)
{
  parent_cv(left_cv, right_cv, key_words, ROOT, out_cv);
}

// BLAKE3 hash function, can be used when chunk count is power of 2
//
// Note, chunk count is preferred to be relatively large number ( say >= 2^20 )
// because this function is supposed to be executed on accelerator
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1876-L2006
void
hash(sycl::queue& q,
     const sycl::uchar* const __restrict input,
     const size_t i_size, // bytes
     const size_t chunk_count,
     sycl::uchar* const __restrict digest,
     sycl::cl_ulong* const __restrict ts)
{
  // whole input byte array is splitted into N -many chunks, each
  // of 1024 -bytes width
  assert(i_size == chunk_count * CHUNK_LEN);
  // minimum 1MB input size for this implementation
  assert(chunk_count >= (1 << 10)); // but you would probably want >= 2^20
  assert((chunk_count & (chunk_count - 1)) == 0); // ensure power of 2

  const size_t mem_size = static_cast<size_t>(BLOCK_LEN) * chunk_count;
  sycl::uint* mem = static_cast<sycl::uint*>(sycl::malloc_device(mem_size, q));

  sycl::event evt =
    q.single_task<kernelBlake3Hash>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] const size_t mem_offset =
        (OUT_LEN >> 2) * chunk_count;

      // compress all chunks so that each chunk produces a single output
      // chaining value of 32 -bytes, which are to be later used for
      // computing parent chaining values ( like Binary Merklization )
#pragma unroll 4
      [[intel::ivdep]]
      for (size_t i = 0; i < chunk_count; i++)
      {
        chunkify(IV,
                 static_cast<sycl::ulong>(i),
                 0,
                 input + i * CHUNK_LEN,
                 mem + mem_offset + i * (OUT_LEN >> 2));
      }

      // `chunk_count` -many leaf nodes of BLAKE3's internal binary merkle tree
      // to be merged in `rounds` -many dispatch rounds such that we reach
      // root node ( i.e. chaining value ) level of merkle tree
      [[intel::fpga_register]] const size_t rounds =
        static_cast<size_t>(sycl::log2(static_cast<double>(chunk_count))) - 1;

      // process each level of nodes of BLAKE3's internal binary merkle tree
      // in order
      //
      // as round 0 must complete before round 1 can begin, I can't let compiler
      // coalesce following nested loop construction
      [[intel::loop_coalesce(1)]] for (size_t r = 0; r < rounds; r++)
      {
        [[intel::fpga_register]] const size_t read_offset = mem_offset >> r;
        [[intel::fpga_register]] const size_t write_offset = read_offset >> 1;
        [[intel::fpga_register]] const size_t parent_count =
          chunk_count >> (r + 1);

        // computing output chaining values of all children nodes of
        // some level of BlAKE3 binary merkle tree
        // can occur in parallel, no data dependency !
        //
        // but ensure that following loop completes execution for (r = 0, see
        // above) before r = 1's body execution can begin, due to presence of
        // critical data dependency !
#pragma unroll 2
        [[intel::ivdep]]
        for (size_t i = 0; i < parent_count; i++)
        {
          parent_cv(mem + read_offset + (i << 1) * (OUT_LEN >> 2),
                    mem + read_offset + ((i << 1) + 1) * (OUT_LEN >> 2),
                    IV,
                    0,
                    mem + write_offset + i * (OUT_LEN >> 2));
        }
      }

      // finally compute root chaining value ( which is digest ) of BLAKE3
      // merkle tree
      root_cv(mem + ((OUT_LEN >> 2) << 1) + 0 * (OUT_LEN >> 2),
              mem + ((OUT_LEN >> 2) << 1) + 1 * (OUT_LEN >> 2),
              IV,
              mem + 1 * (OUT_LEN >> 2));
      // write 32 -bytes BLAKE3 digest back to allocated memory !
      words_to_le_bytes(mem + 1 * (OUT_LEN >> 2), digest);
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
