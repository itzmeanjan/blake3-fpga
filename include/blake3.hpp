#pragma once
#define SYCL_SIMPLE_SWIZZLES 1
#include <CL/sycl.hpp>

namespace blake3 {

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
rnd(sycl::uint4* const state, const sycl::uint* msg_words)
{
  const sycl::uint4 mx = sycl::uint4(
    *(msg_words + 0), *(msg_words + 2), *(msg_words + 4), *(msg_words + 6));
  const sycl::uint4 my = sycl::uint4(
    *(msg_words + 1), *(msg_words + 3), *(msg_words + 5), *(msg_words + 7));
  const sycl::uint4 mz = sycl::uint4(
    *(msg_words + 8), *(msg_words + 10), *(msg_words + 12), *(msg_words + 14));
  const sycl::uint4 mw = sycl::uint4(
    *(msg_words + 9), *(msg_words + 11), *(msg_words + 13), *(msg_words + 15));

  constexpr sycl::uint4 rrot_16 = sycl::uint4(16); // = 32 - 16
  constexpr sycl::uint4 rrot_12 = sycl::uint4(20); // = 32 - 12
  constexpr sycl::uint4 rrot_8 = sycl::uint4(24);  // = 32 - 8
  constexpr sycl::uint4 rrot_7 = sycl::uint4(25);  // = 32 - 7

  // column-wise mixing
  *(state + 0) = *(state + 0) + *(state + 1) + mx;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_16);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_12);
  *(state + 0) = *(state + 0) + *(state + 1) + my;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_8);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // diagonalize
  *(state + 1) = (*(state + 1)).yzwx();
  *(state + 2) = (*(state + 2)).zwxy();
  *(state + 3) = (*(state + 3)).wxyz();

  // diagonal mixing
  *(state + 0) = *(state + 0) + *(state + 1) + mz;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_16);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_12);
  *(state + 0) = *(state + 0) + *(state + 1) + mw;
  *(state + 3) = sycl::rotate(*(state + 3) ^ *(state + 0), rrot_8);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = sycl::rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // un-diagonalize
  *(state + 1) = (*(state + 1)).wxyz();
  *(state + 2) = (*(state + 2)).zwxy();
  *(state + 3) = (*(state + 3)).yzwx();
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
  sycl::uint permuted[16];

#pragma unroll 16 // fully unroll this loop
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = *(msg_words + MSG_PERMUTATION[i]);
  }

#pragma unroll 16 // fully unroll this loop
  for (size_t i = 0; i < 16; i++) {
    *(msg_words + i) = permuted[i];
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
         sycl::uint* const block_words,
         sycl::ulong counter,
         sycl::uint block_len,
         sycl::uint flags,
         sycl::uint* const out_cv)
{
  sycl::uint4 state[4] = {
    sycl::uint4(*(in_cv + 0), *(in_cv + 1), *(in_cv + 2), *(in_cv + 3)),
    sycl::uint4(*(in_cv + 4), *(in_cv + 5), *(in_cv + 6), *(in_cv + 7)),
    sycl::uint4(IV[0], IV[1], IV[2], IV[3]),
    sycl::uint4(counter & 0xffffffff,
                static_cast<sycl::uint>(counter >> 32),
                block_len,
                flags)
  };

  // round 1
  rnd(state, block_words);
  permute(block_words);

  // round 2
  rnd(state, block_words);
  permute(block_words);

  // round 3
  rnd(state, block_words);
  permute(block_words);

  // round 4
  rnd(state, block_words);
  permute(block_words);

  // round 5
  rnd(state, block_words);
  permute(block_words);

  // round 6
  rnd(state, block_words);
  permute(block_words);

  // round 7
  rnd(state, block_words);
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

}
