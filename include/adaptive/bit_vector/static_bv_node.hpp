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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"

namespace adaptive {

template <uint32_t BlockFactor = 4> class static_bv_node {
  public:
    static constexpr uint32_t block_factor = BlockFactor;
    static constexpr uint32_t word_bits = 64;
    static constexpr uint32_t w16 = 8 * sizeof(uint16_t);
    static constexpr uint32_t superblock_bits = 1 << w16;
    static constexpr uint32_t block_bits = block_factor * word_bits;

    static_assert(block_factor > 0);
    static_assert(superblock_bits % block_bits == 0);

    static_bv_node(const uint64_t* words, uint64_t size);
    static_bv_node(std::vector<uint64_t>&& words, uint64_t size);

    uint64_t size() const;
    uint64_t ones() const;
    uint64_t space_bytes() const;
    const uint64_t* data() const;

    bool access(uint64_t i) const;
    void read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit) const;

    template <bool Bit> uint64_t rank(uint64_t i) const;
    template <bool Bit> uint64_t select(uint64_t j) const;
    template <bool Bit> int64_t next(uint64_t i) const;

  private:
    uint64_t size_ = 0;
    uint64_t ones_ = 0;
    std::vector<uint64_t> words_;
    std::vector<uint64_t> superblocks_;
    std::vector<uint16_t> blocks_;

    uint64_t word_count() const;
    uint64_t block_count() const;
    uint64_t superblock_count() const;

    template <bool Bit> uint64_t total_bits() const;
    template <bool Bit> uint64_t superblock_prefix(uint64_t superblock) const;
    template <bool Bit> uint64_t block_prefix(uint64_t block, uint64_t superblock_first_block) const;
    template <bool Bit> uint64_t word_bits_at(uint64_t word) const;

    void preprocess();
};

template <uint32_t BlockFactor>
static_bv_node<BlockFactor>::static_bv_node(const uint64_t* words, uint64_t size) : size_(size) {
    ADAPTIVE_ASSUME(size > 0);

    words_.assign(words, words + word_count());
    if (size_ % word_bits != 0) {
        words_.back() &= (uint64_t{1} << (size_ % word_bits)) - 1;
    }
    preprocess();
}

template <uint32_t BlockFactor>
static_bv_node<BlockFactor>::static_bv_node(std::vector<uint64_t>&& words, uint64_t size)
    : size_(size), words_(std::move(words)) {
    ADAPTIVE_ASSUME(size > 0);

    words_.resize(word_count());
    if (size_ % word_bits != 0) {
        words_.back() &= (uint64_t{1} << (size_ % word_bits)) - 1;
    }
    preprocess();
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::size() const {
    return size_;
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::ones() const {
    return ones_;
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::space_bytes() const {
    return sizeof(*this) + words_.capacity() * sizeof(uint64_t) + superblocks_.capacity() * sizeof(uint64_t) +
           blocks_.capacity() * sizeof(uint16_t);
}

template <uint32_t BlockFactor> const uint64_t* static_bv_node<BlockFactor>::data() const {
    return words_.data();
}

template <uint32_t BlockFactor> bool static_bv_node<BlockFactor>::access(uint64_t i) const {
    ADAPTIVE_ASSUME(i < size_);

    return (words_[i / word_bits] >> (i % word_bits)) & 1;
}

template <uint32_t BlockFactor>
void static_bv_node<BlockFactor>::read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit) const {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(len <= size_ - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);

    bit_util::copy_bits(out, out_bit, words_.data(), i, len);
}

template <uint32_t BlockFactor> template <bool Bit> uint64_t static_bv_node<BlockFactor>::rank(uint64_t i) const {
    ADAPTIVE_ASSUME(i <= size_);

    if constexpr (!Bit) {
        return i - rank<1>(i);
    }

    if (i == size_) {
        return ones_;
    }

    const uint64_t block = i / block_bits;
    uint64_t rank = superblocks_[i >> w16] + blocks_[block];
    const uint64_t first_word = block * block_factor;
    const uint64_t full_words = i / word_bits;
    for (uint64_t word = first_word; word < full_words; ++word) {
        rank += bit_util::popcount(words_[word]);
    }

    const uint64_t bits = i % word_bits;
    if (bits == 0) {
        return rank;
    }
    const uint64_t mask = (uint64_t{1} << bits) - 1;
    return rank + bit_util::popcount(words_[full_words] & mask);
}

template <uint32_t BlockFactor> template <bool Bit> uint64_t static_bv_node<BlockFactor>::select(uint64_t j) const {
    const uint64_t t = total_bits<Bit>();

    ADAPTIVE_ASSUME(j > 0);
    ADAPTIVE_ASSUME(j <= t);

    int64_t d;
    uint64_t m;
    const uint64_t n = size_;
    uint64_t s = superblock_count();

    int64_t i = static_cast<int64_t>((j * s) / t);
    if (i == static_cast<int64_t>(s)) {
        i--;
    }
    uint64_t pref = superblock_prefix<Bit>(i);
    if (pref < j) {
        d = 1;
        while (i + d < static_cast<int64_t>(s)) {
            pref = superblock_prefix<Bit>(i + d);
            if (pref >= j) {
                break;
            }
            i += d;
            d <<= 1;
        }
        d = std::min(static_cast<int64_t>(s), i + d);
        while (i + 1 < d) {
            m = (i + d) >> 1;
            pref = superblock_prefix<Bit>(m);
            if (pref < j) {
                i = m;
            } else {
                d = m;
            }
        }
    } else {
        d = 1;
        while (i - d >= 0) {
            pref = superblock_prefix<Bit>(i - d);
            if (pref < j) {
                break;
            }
            i -= d;
            d <<= 1;
        }
        d = std::max<int64_t>(0, i - d);
        while (d + 1 < i) {
            m = (i + d) >> 1;
            pref = superblock_prefix<Bit>(m);
            if (pref < j) {
                d = m;
            } else {
                i = m;
            }
        }
        i--;
    }

    pref = superblock_prefix<Bit>(i);
    j -= pref;

    uint64_t p;
    if (i < static_cast<int64_t>(s - 1)) {
        p = superblock_prefix<Bit>(i + 1) - superblock_prefix<Bit>(i);
    } else {
        p = t - superblock_prefix<Bit>(i);
    }

    const int64_t b = (i << w16) / block_bits;
    s = std::min<uint64_t>(b + superblock_bits / block_bits, block_count());
    i = b + static_cast<int64_t>((j * (s - b)) / p);
    if (i == static_cast<int64_t>(s)) {
        i--;
    }
    pref = block_prefix<Bit>(i, b);
    if (pref < j) {
        d = 1;
        while (i + d < static_cast<int64_t>(s)) {
            pref = block_prefix<Bit>(i + d, b);
            if (pref >= j) {
                break;
            }
            i += d;
            d <<= 1;
        }
        d = std::min(static_cast<int64_t>(s), i + d);
        while (i + 1 < d) {
            m = (i + d) >> 1;
            pref = block_prefix<Bit>(m, b);
            if (pref < j) {
                i = m;
            } else {
                d = m;
            }
        }
    } else {
        d = 1;
        while (i - d >= b) {
            pref = block_prefix<Bit>(i - d, b);
            if (pref < j) {
                break;
            }
            i -= d;
            d <<= 1;
        }
        d = std::max<int64_t>(b, i - d);
        while (d + 1 < i) {
            m = (i + d) >> 1;
            pref = block_prefix<Bit>(m, b);
            if (pref < j) {
                d = m;
            } else {
                i = m;
            }
        }
        i--;
    }

    pref = block_prefix<Bit>(i, b);
    j -= pref;
    i *= block_factor;
    while (static_cast<uint64_t>(i + 1) * word_bits < n) {
        const uint64_t word = word_bits_at<Bit>(i);
        p = bit_util::popcount(word);
        if (p >= j) {
            break;
        }
        j -= p;
        i++;
    }
    uint64_t word = word_bits_at<Bit>(i);
    if (static_cast<uint64_t>(i + 1) * word_bits > n) {
        word &= (uint64_t{1} << (n % word_bits)) - 1;
    }
    i *= word_bits;
    return i + bit_util::select_in_word(word, j);
}

template <uint32_t BlockFactor> template <bool Bit> int64_t static_bv_node<BlockFactor>::next(uint64_t i) const {
    ADAPTIVE_ASSUME(i < size_);

    uint64_t p = i / word_bits;
    uint64_t word = word_bits_at<Bit>(p);
    word &= ~uint64_t{0} << (i % word_bits);
    if ((p + 1) * word_bits > size_) {
        word &= (uint64_t{1} << (size_ % word_bits)) - 1;
    }
    if (word) {
        return p * word_bits + bit_util::next_in_word(word);
    }

    const uint64_t b = std::min((p / block_factor + 2) * block_factor, 1 + (size_ - 1) / word_bits);
    p++;
    while (p < b) {
        word = word_bits_at<Bit>(p++);
        if (word) {
            break;
        }
    }
    if (word) {
        if (p * word_bits > size_) {
            word &= (uint64_t{1} << (size_ % word_bits)) - 1;
        }
        if (!word) {
            return -1;
        }
        return (p - 1) * word_bits + bit_util::next_in_word(word);
    } else if (p == 1 + (size_ - 1) / word_bits) {
        return -1;
    }

    const uint64_t sb = p / block_factor - 1;
    const uint64_t s = (sb * block_bits) >> w16;
    const uint64_t fb = s * (superblock_bits / block_bits);
    const uint64_t rank = superblock_prefix<Bit>(s) + block_prefix<Bit>(sb, fb);
    const uint64_t t = total_bits<Bit>();
    if (rank == t) {
        return -1;
    }
    return select<Bit>(rank + 1);
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::word_count() const {
    return (size_ + word_bits - 1) / word_bits;
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::block_count() const {
    return (size_ + block_bits - 1) / block_bits;
}

template <uint32_t BlockFactor> uint64_t static_bv_node<BlockFactor>::superblock_count() const {
    return (size_ + superblock_bits - 1) / superblock_bits;
}

template <uint32_t BlockFactor> template <bool Bit> uint64_t static_bv_node<BlockFactor>::total_bits() const {
    if constexpr (Bit) {
        return ones_;
    } else {
        return size_ - ones_;
    }
}

template <uint32_t BlockFactor>
template <bool Bit>
uint64_t static_bv_node<BlockFactor>::superblock_prefix(uint64_t superblock) const {
    if constexpr (Bit) {
        return superblocks_[superblock];
    } else {
        return superblock * superblock_bits - superblocks_[superblock];
    }
}

template <uint32_t BlockFactor>
template <bool Bit>
uint64_t static_bv_node<BlockFactor>::block_prefix(uint64_t block, uint64_t superblock_first_block) const {
    if constexpr (Bit) {
        return blocks_[block];
    } else {
        return (block - superblock_first_block) * block_bits - blocks_[block];
    }
}

template <uint32_t BlockFactor>
template <bool Bit>
uint64_t static_bv_node<BlockFactor>::word_bits_at(uint64_t word) const {
    if constexpr (Bit) {
        return words_[word];
    } else {
        return ~words_[word];
    }
}

template <uint32_t BlockFactor> void static_bv_node<BlockFactor>::preprocess() {
    blocks_.resize(block_count());
    superblocks_.resize(superblock_count());

    uint64_t acc = 0;
    uint64_t sacc = 0;
    uint64_t i = 0;
    while (i < word_count()) {
        const uint64_t top = std::min(word_count(), i + superblock_bits / word_bits);
        sacc += acc;
        acc = 0;
        superblocks_[(i * word_bits) >> w16] = sacc;
        while (i < top) {
            if (i % block_factor == 0) {
                blocks_[i / block_factor] = acc;
            }
            acc += bit_util::popcount(words_[i]);
            i++;
        }
    }
    ones_ = sacc + acc;
}

} // namespace adaptive
