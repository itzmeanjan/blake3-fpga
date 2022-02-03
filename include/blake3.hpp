#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>
#include <cassert>

namespace blake3 {

// just to avoid kernel name mangling issue in optimization report
class kernelBlake3HashChunkifyLeafNodes;
class kernelBlake3HashParentChaining;
class kernelBlake3HashRootChaining;

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

// BLAKE3 round function, applied 7 times for mixing 64 -bytes input message
// words ( each message word = 32 -bit ) into hash state
//
// During each round, first eight message words are mixed into hash state
// column wise and then remaining eight message words are mixed diagonally.
// For that reason 4x4 hash state matrix needs to be diagonalised after
// column-wise mixing ( so that diagonal mixing can be applied using SYCL
// vector intrinsics, as applied during column-wise mixing ) and finally
// un-diagonalized ( preparing for next mixing round ) after diagonal mixing
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1569-L1621
inline void
rnd(sycl::uint4* const state, const sycl::uint* msg)
{
  [[intel::fpga_memory]] const sycl::uint4 mx =
    sycl::uint4(msg[0], msg[2], msg[4], msg[6]);
  [[intel::fpga_memory]] const sycl::uint4 my =
    sycl::uint4(msg[1], msg[3], msg[5], msg[7]);
  [[intel::fpga_memory]] const sycl::uint4 mz =
    sycl::uint4(msg[8], msg[10], msg[12], msg[14]);
  [[intel::fpga_memory]] const sycl::uint4 mw =
    sycl::uint4(msg[9], msg[11], msg[13], msg[15]);

  [[intel::fpga_memory]] constexpr sycl::uint4 rrot_16 =
    sycl::uint4(16); // = 32 - 16
  [[intel::fpga_memory]] constexpr sycl::uint4 rrot_12 =
    sycl::uint4(20); // = 32 - 12
  [[intel::fpga_memory]] constexpr sycl::uint4 rrot_8 =
    sycl::uint4(24); // = 32 - 8
  [[intel::fpga_memory]] constexpr sycl::uint4 rrot_7 =
    sycl::uint4(25); // = 32 - 7

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
permute(sycl::uint* const msg_words)
{
  // additional array memory ( = 64 -bytes ) for helping in message word
  // permutation
  [[intel::fpga_memory]] sycl::uint permuted[16];

#pragma unroll 16 // fully unroll this loop
  [[intel::ivdep]]
  for (size_t i = 0; i < 16; i++)
  {
    permuted[i] = msg_words[MSG_PERMUTATION[i]];
  }

#pragma unroll 16 // fully unroll this loop
  [[intel::ivdep]]
  for (size_t i = 0; i < 16; i++)
  {
    msg_words[i] = permuted[i];
  }
}

// BLAKE3 compression function, which mixes 64 -bytes message block into
// 4x4 hash state matrix ( which is also 64 -bytes, because word size of BLAKE3
// is 32 -bit ) by 7 mixing rounds and 6 permutation rounds
//
// See
// https://github.com/itzmeanjan/blake3/blob/f07d32ec10cbc8a10663b7e6539e0b1dab3e453b/include/blake3.hpp#L1641-L1703
inline void
compress(const sycl::uint* in_cv,
         sycl::uint* const msg_words,
         const sycl::ulong counter,
         const sycl::uint block_len,
         const sycl::uint flags,
         sycl::uint* const out_cv)
{
  [[intel::fpga_memory]] sycl::uint4 state[4] = {
    sycl::uint4(in_cv[0], in_cv[1], in_cv[2], in_cv[3]),
    sycl::uint4(in_cv[4], in_cv[5], in_cv[6], in_cv[7]),
    sycl::uint4(IV[0], IV[1], IV[2], IV[3]),
    sycl::uint4(static_cast<sycl::uint>(counter & 0xffffffff),
                static_cast<sycl::uint>(counter >> 32),
                block_len,
                flags)
  };

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

  state[0] ^= state[2];
  state[1] ^= state[3];
  // following two lines don't dictate output chaining value
  // of this block ( or chunk ), so they can be safely commented out !
  // state[2] ^= cv0;
  // state[3] ^= cv1;
  // see
  // https://github.com/BLAKE3-team/BLAKE3/blob/da4c792/reference_impl/reference_impl.rs#L118

  // output chaining value of this block to be used as
  // input chaining value for next block in same chunk
  {
    auto priv_out_cv = sycl::private_ptr<sycl::uint>(out_cv);
    state[0].store(0, priv_out_cv);
    state[1].store(1, priv_out_cv);
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
  [[intel::ivdep]]
  for (size_t i = 0; i < 16; i++)
  {
    msg_words[i] = word_from_le_bytes(input + (i << 2));
  }
}

// One 32 -bit BLAKE3 word is converted to four consecutive little endian bytes
static inline void
word_to_le_bytes(const sycl::uint word, sycl::uchar* const output)
{
 #pragma unroll 4
  [[intel::ivdep]]
  for(size_t i = 0; i < 4; i++) {
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
  [[intel::ivdep]]
  for (size_t i = 0; i < 8; i++)
  {
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
  [[intel::fpga_memory]] sycl::uint in_cv[8];
  [[intel::fpga_memory]] sycl::uint priv_out_cv[8];
  [[intel::fpga_memory]] sycl::uint msg_words[16];

#pragma unroll 8 // attempt to fully parallelize array initialization !
  [[intel::ivdep]]
  for (size_t i = 0; i < 8; i++)
  {
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
  [[intel::ivdep]]
  for (size_t j = 0; j < 8; j++)
  {
    in_cv[j] = priv_out_cv[j];
  }
  // --- end processing first message block ---

  // process intermediate ( read non-boundary ) 14 message blocks
  for (size_t i = 1; i < 15; i++)
  {
    words_from_le_bytes(input + i * BLOCK_LEN, msg_words);
    compress(in_cv, msg_words, chunk_counter, BLOCK_LEN, flags, priv_out_cv);

#pragma unroll 8 // copying between array can be fully parallelized !
    [[intel::ivdep]]
    for (size_t j = 0; j < 8; j++)
    {
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
  [[intel::fpga_memory]] sycl::uint block_words[16];

#pragma unroll 8
  [[intel::ivdep]]
  for (size_t i = 0; i < 8; i++)
  {
    block_words[i] = left_cv[i];
  }
#pragma unroll 8
  [[intel::ivdep]]
  for (size_t i = 0; i < 8; i++)
  {
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
     sycl::uchar* const __restrict digest)
{
  assert(i_size == chunk_count * CHUNK_LEN);
  // minimum 1MB input size for this implementation
  assert(chunk_count >= (1 << 10)); // but you would probably want >= 2^20
  assert((chunk_count & (chunk_count - 1)) == 0); // ensure power of 2

  const size_t mem_size = static_cast<size_t>(BLOCK_LEN) * chunk_count;
  sycl::uint* mem = static_cast<sycl::uint*>(sycl::malloc_device(mem_size, q));
  const size_t mem_offset = (OUT_LEN >> 2) * chunk_count;

  sycl::event evt_0 = q.single_task<kernelBlake3HashChunkifyLeafNodes>([=
  ]() [[intel::kernel_args_restrict]] {
    [[intel::ivdep]]
    for (size_t i = 0; i < chunk_count; i++)
    {
      chunkify(IV,
               static_cast<sycl::ulong>(i),
               0,
               input + i * CHUNK_LEN,
               mem + mem_offset + i * (OUT_LEN >> 2));
    }
  });

  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(chunk_count))) - 1;

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      const size_t read_offset = mem_offset >> r;
      const size_t write_offset = read_offset >> 1;
      const size_t glb_work_items = chunk_count >> (r + 1);

      h.single_task<kernelBlake3HashParentChaining>([=
      ]() [[intel::kernel_args_restrict]] {
        [[intel::ivdep]]
        for (size_t i = 0; i < glb_work_items; i++)
        {
          parent_cv(mem + read_offset + (i << 1) * (OUT_LEN >> 2),
                    mem + read_offset + ((i << 1) + 1) * (OUT_LEN >> 2),
                    IV,
                    0,
                    mem + write_offset + i * (OUT_LEN >> 2));
        }
      });
    });
    evts.push_back(evt);
  }

  sycl::event evt_1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(rounds - 1));
    h.single_task<kernelBlake3HashRootChaining>([=
    ]() [[intel::kernel_args_restrict]] {
      root_cv(mem + ((OUT_LEN >> 2) << 1) + 0 * (OUT_LEN >> 2),
              mem + ((OUT_LEN >> 2) << 1) + 1 * (OUT_LEN >> 2),
              IV,
              mem + 1 * (OUT_LEN >> 2));
      words_to_le_bytes(mem + 1 * (OUT_LEN >> 2), digest);
    });
  });

  evt_1.wait();
  sycl::free(mem, q);
}

}
