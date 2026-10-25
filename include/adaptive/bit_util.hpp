/*

adaptive -- an implementation of adaptive dynamic bitvectors and packed arrays.
Copyright (C) 2024 Gonzalo Navarro
Copyright (C) 2026 Diego Arias

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

Author's contact: Gonzalo Navarro, Dept. of Computer Science, University of
Chile. Beauchef 851, Santiago, Chile. gnavarro@dcc.uchile.cl

*/

#pragma once

#include <cstdint>
#include <cstring>
#if defined(__BMI2__)
#include <immintrin.h>
#endif

#include "adaptive/assert.hpp"

namespace adaptive::bit_util {

static constexpr uint32_t word_bits = 64;

inline uint32_t popcount(uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(word);
#else
    word -= (word >> 1) & 0x5555555555555555ULL;
    word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
    return (((word + (word >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56;
#endif
}

inline uint32_t select_in_word(uint64_t word, uint32_t rank);

inline uint32_t next_in_word(uint64_t word) {
    return select_in_word(word, 1);
}

namespace detail {

inline constexpr uint8_t lt_sel[] = {
    0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2,
    0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0,
    1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
    0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
    2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
    0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
    1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,

    0, 0, 0, 1, 0, 2, 2, 1, 0, 3, 3, 1, 3, 2, 2, 1, 0, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 0, 5, 5, 1, 5,
    2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 0, 6, 6, 1, 6, 2, 2, 1, 6, 3,
    3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2,
    1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 0, 7, 7, 1, 7, 2, 2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1,
    4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4,
    3, 3, 1, 3, 2, 2, 1, 7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
    2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,

    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 3, 2, 0, 0, 0, 4, 0, 4, 4, 2, 0, 4, 4, 3, 4, 3, 3, 2, 0, 0, 0, 5, 0,
    5, 5, 2, 0, 5, 5, 3, 5, 3, 3, 2, 0, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 0, 0, 0, 6, 0, 6, 6, 2, 0, 6,
    6, 3, 6, 3, 3, 2, 0, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 0, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3,
    2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 0, 0, 0, 7, 0, 7, 7, 2, 0, 7, 7, 3, 7, 3, 3, 2, 0, 7, 7, 4,
    7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 0, 7, 7, 5, 7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5,
    4, 4, 3, 4, 3, 3, 2, 0, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3,
    3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 3, 0, 0, 0, 0, 0,
    0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 3, 0, 0, 0, 5, 0, 5, 5, 4, 0, 5, 5, 4, 5, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
    0, 6, 0, 6, 6, 3, 0, 0, 0, 6, 0, 6, 6, 4, 0, 6, 6, 4, 6, 4, 4, 3, 0, 0, 0, 6, 0, 6, 6, 5, 0, 6, 6, 5, 6, 5, 5,
    3, 0, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 3, 0, 0, 0, 7,
    0, 7, 7, 4, 0, 7, 7, 4, 7, 4, 4, 3, 0, 0, 0, 7, 0, 7, 7, 5, 0, 7, 7, 5, 7, 5, 5, 3, 0, 7, 7, 5, 7, 5, 5, 4, 7,
    5, 5, 4, 5, 4, 4, 3, 0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 3, 0, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4, 6, 4,
    4, 3, 0, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6,
    5, 0, 0, 0, 6, 0, 6, 6, 5, 0, 6, 6, 5, 6, 5, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 5, 0, 0, 0, 7, 0, 7, 7, 5, 0,
    7, 7, 5, 7, 5, 5, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6, 0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6,
    6, 4, 0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 5, 0, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0,
    0, 0, 7, 0, 7, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7,
    7, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6, 0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 5,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7};

inline constexpr uint64_t ps_overflow[] = {
    0x8080808080808080ULL, 0x7f7f7f7f7f7f7f7fULL, 0x7e7e7e7e7e7e7e7eULL, 0x7d7d7d7d7d7d7d7dULL, 0x7c7c7c7c7c7c7c7cULL,
    0x7b7b7b7b7b7b7b7bULL, 0x7a7a7a7a7a7a7a7aULL, 0x7979797979797979ULL, 0x7878787878787878ULL, 0x7777777777777777ULL,
    0x7676767676767676ULL, 0x7575757575757575ULL, 0x7474747474747474ULL, 0x7373737373737373ULL, 0x7272727272727272ULL,
    0x7171717171717171ULL, 0x7070707070707070ULL, 0x6f6f6f6f6f6f6f6fULL, 0x6e6e6e6e6e6e6e6eULL, 0x6d6d6d6d6d6d6d6dULL,
    0x6c6c6c6c6c6c6c6cULL, 0x6b6b6b6b6b6b6b6bULL, 0x6a6a6a6a6a6a6a6aULL, 0x6969696969696969ULL, 0x6868686868686868ULL,
    0x6767676767676767ULL, 0x6666666666666666ULL, 0x6565656565656565ULL, 0x6464646464646464ULL, 0x6363636363636363ULL,
    0x6262626262626262ULL, 0x6161616161616161ULL, 0x6060606060606060ULL, 0x5f5f5f5f5f5f5f5fULL, 0x5e5e5e5e5e5e5e5eULL,
    0x5d5d5d5d5d5d5d5dULL, 0x5c5c5c5c5c5c5c5cULL, 0x5b5b5b5b5b5b5b5bULL, 0x5a5a5a5a5a5a5a5aULL, 0x5959595959595959ULL,
    0x5858585858585858ULL, 0x5757575757575757ULL, 0x5656565656565656ULL, 0x5555555555555555ULL, 0x5454545454545454ULL,
    0x5353535353535353ULL, 0x5252525252525252ULL, 0x5151515151515151ULL, 0x5050505050505050ULL, 0x4f4f4f4f4f4f4f4fULL,
    0x4e4e4e4e4e4e4e4eULL, 0x4d4d4d4d4d4d4d4dULL, 0x4c4c4c4c4c4c4c4cULL, 0x4b4b4b4b4b4b4b4bULL, 0x4a4a4a4a4a4a4a4aULL,
    0x4949494949494949ULL, 0x4848484848484848ULL, 0x4747474747474747ULL, 0x4646464646464646ULL, 0x4545454545454545ULL,
    0x4444444444444444ULL, 0x4343434343434343ULL, 0x4242424242424242ULL, 0x4141414141414141ULL, 0x4040404040404040ULL};

} // namespace detail

inline uint32_t select_in_word(uint64_t word, uint32_t rank) {
    ADAPTIVE_ASSUME(rank > 0);
    ADAPTIVE_ASSUME(rank <= popcount(word));

#if defined(__BMI2__)
    return __builtin_ctzll(_pdep_u64(uint64_t{1} << (rank - 1), word));
#else
    uint64_t i = rank;
    uint64_t sums = word - ((word >> 1) & 0x5555555555555555ULL);
    sums = (sums & 0x3333333333333333ULL) + ((sums >> 2) & 0x3333333333333333ULL);
    sums = (sums + (sums >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    sums = 0x0101010101010101ULL * sums;

    const uint64_t overflow = (sums + detail::ps_overflow[i]) & 0x8080808080808080ULL;
    const uint32_t byte_index = __builtin_ctzll(overflow) >> 3;

    sums <<= 8;
    i -= (sums >> (byte_index << 3)) & 0xFFULL;
    return (byte_index << 3) + detail::lt_sel[((i - 1) << 8) + ((word >> (byte_index << 3)) & 0xFFULL)];
#endif
}

inline void copy_bits(uint64_t* target, uint64_t target_bit, const uint64_t* source, uint64_t source_bit,
                      uint64_t len) {
    if (len == 0) {
        return;
    }

    target += target_bit / word_bits;
    target_bit %= word_bits;
    source += source_bit / word_bits;
    source_bit %= word_bits;

    const uint64_t target_mask = (uint64_t{1} << target_bit) - 1;
    if (target_bit == source_bit) {
        if (target_bit != 0) {
            *target = (*target & target_mask) | (*source & ~target_mask);
            ++target;
            ++source;
            if (len <= word_bits - target_bit) {
                return;
            }
            len -= word_bits - target_bit;
        }
        std::memcpy(target, source, ((len + word_bits - 1) / word_bits) * sizeof(uint64_t));
        return;
    }

    if (target_bit < source_bit) {
        *target = (*target & target_mask) | ((*source++ >> (source_bit - target_bit)) & ~target_mask);
        if (len <= word_bits - source_bit) {
            return;
        }
        target_bit += word_bits - source_bit;
    } else {
        *target = (*target & target_mask) | ((*source << (target_bit - source_bit)) & ~target_mask);
        if (len <= word_bits - target_bit) {
            return;
        }
        target_bit -= source_bit;
        *++target = *source++ >> (word_bits - target_bit);
    }

    if (len <= word_bits - source_bit) {
        return;
    }
    len -= word_bits - source_bit;

    const uint64_t aligned_mask = (uint64_t{1} << target_bit) - 1;
    uint64_t old = *target & aligned_mask;
    len += word_bits;
    while (len > word_bits) {
        *target++ = old | (*source << target_bit);
        old = *source++ >> (word_bits - target_bit);
        len -= word_bits;
    }
    if (len + target_bit > word_bits) {
        *target = old;
    }
}

} // namespace adaptive::bit_util
