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

#include <array>
#include <cstdint>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"

namespace adaptive {

template <uint32_t MaxLeafBits = 8192> class leaf_bv_node {
  public:
    static constexpr uint32_t word_bits = 64;
    static constexpr uint32_t max_bits = MaxLeafBits;
    static_assert(max_bits % word_bits == 0);
    static constexpr uint32_t max_word_count = max_bits / word_bits;

    leaf_bv_node() = default;
    leaf_bv_node(const uint64_t* words, uint32_t size);
    leaf_bv_node(const uint64_t* words, uint64_t offset, uint32_t size);

    uint32_t size() const;
    uint32_t ones() const;
    uint64_t space_bytes() const;
    const uint64_t* data() const;

    bool access(uint32_t i) const;
    void read(uint32_t i, uint32_t len, uint64_t* out, uint64_t out_bit) const;

    template <bool Bit> uint32_t rank(uint32_t i) const;
    template <bool Bit> uint32_t select(uint32_t j) const;
    template <bool Bit> int32_t next(uint32_t i) const;

    int32_t write(uint32_t i, bool value);
    void insert(uint32_t i, bool value);
    int32_t erase(uint32_t i);

  private:
    uint32_t size_ = 0;
    uint32_t ones_ = 0;
    std::array<uint64_t, max_word_count> words_{};

    uint32_t word_count() const;
};

template <uint32_t MaxLeafBits>
leaf_bv_node<MaxLeafBits>::leaf_bv_node(const uint64_t* words, uint32_t size) : leaf_bv_node(words, 0, size) {}

template <uint32_t MaxLeafBits>
leaf_bv_node<MaxLeafBits>::leaf_bv_node(const uint64_t* words, uint64_t offset, uint32_t size) : size_(size) {
    ADAPTIVE_ASSUME(size <= max_bits);
    ADAPTIVE_ASSUME(words != nullptr || size == 0);

    if (size_ != 0) {
        bit_util::copy_bits(words_.data(), 0, words, offset, size_);
    }
    for (uint32_t i = 0; i < word_count(); ++i) {
        ones_ += bit_util::popcount(words_[i]);
    }
}

template <uint32_t MaxLeafBits> uint32_t leaf_bv_node<MaxLeafBits>::size() const {
    return size_;
}

template <uint32_t MaxLeafBits> uint32_t leaf_bv_node<MaxLeafBits>::ones() const {
    return ones_;
}

template <uint32_t MaxLeafBits> uint64_t leaf_bv_node<MaxLeafBits>::space_bytes() const {
    return sizeof(*this);
}

template <uint32_t MaxLeafBits> const uint64_t* leaf_bv_node<MaxLeafBits>::data() const {
    return words_.data();
}

template <uint32_t MaxLeafBits> bool leaf_bv_node<MaxLeafBits>::access(uint32_t i) const {
    ADAPTIVE_ASSUME(i < size_);

    return (words_[i / word_bits] >> (i % word_bits)) & 1;
}

template <uint32_t MaxLeafBits>
void leaf_bv_node<MaxLeafBits>::read(uint32_t i, uint32_t len, uint64_t* out, uint64_t out_bit) const {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(len <= size_ - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);

    bit_util::copy_bits(out, out_bit, words_.data(), i, len);
}

template <uint32_t MaxLeafBits> template <bool Bit> uint32_t leaf_bv_node<MaxLeafBits>::rank(uint32_t i) const {
    ADAPTIVE_ASSUME(i <= size_);

    if constexpr (!Bit) {
        return i - rank<1>(i);
    }

    uint32_t result = 0;
    const uint32_t full_words = i / word_bits;
    for (uint32_t word = 0; word < full_words; ++word) {
        result += bit_util::popcount(words_[word]);
    }

    const uint32_t bits = i % word_bits;
    if (bits == 0) {
        return result;
    }
    const uint64_t mask = (uint64_t{1} << bits) - 1;
    return result + bit_util::popcount(words_[full_words] & mask);
}

template <uint32_t MaxLeafBits> template <bool Bit> uint32_t leaf_bv_node<MaxLeafBits>::select(uint32_t j) const {
    uint32_t total;
    if constexpr (Bit) {
        total = ones_;
    } else {
        total = size_ - ones_;
    }

    ADAPTIVE_ASSUME(j > 0);
    ADAPTIVE_ASSUME(j <= total);

    uint32_t k = 0;
    const uint32_t words = word_count();
    for (uint32_t word = 0; word < words; ++word) {
        uint64_t bits;
        if constexpr (Bit) {
            bits = words_[word];
        } else {
            bits = ~words_[word];
        }
        if (word + 1 == words && size_ % word_bits != 0) {
            bits &= (uint64_t{1} << (size_ % word_bits)) - 1;
        }

        const uint32_t count = bit_util::popcount(bits);
        if (j <= k + count) {
            return word * word_bits + bit_util::select_in_word(bits, j - k);
        }
        k += count;
    }
    ADAPTIVE_UNREACHABLE();
}

template <uint32_t MaxLeafBits> template <bool Bit> int32_t leaf_bv_node<MaxLeafBits>::next(uint32_t i) const {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t first_word = i / word_bits;
    const uint32_t last_word = (size_ - 1) / word_bits;
    for (uint32_t word = first_word; word <= last_word; ++word) {
        uint64_t bits;
        if constexpr (Bit) {
            bits = words_[word];
        } else {
            bits = ~words_[word];
        }
        if (word == first_word) {
            bits &= ~uint64_t{0} << (i % word_bits);
        }
        if (word == last_word && size_ % word_bits != 0) {
            bits &= (uint64_t{1} << (size_ % word_bits)) - 1;
        }
        if (bits != 0) {
            return static_cast<int32_t>(word * word_bits + bit_util::next_in_word(bits));
        }
    }
    return -1;
}

template <uint32_t MaxLeafBits> int32_t leaf_bv_node<MaxLeafBits>::write(uint32_t i, bool value) {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t word = i / word_bits;
    const uint64_t mask = uint64_t{1} << (i % word_bits);
    const bool old_value = (words_[word] & mask) != 0;
    if (old_value == value) {
        return 0;
    }

    if (value) {
        words_[word] |= mask;
    } else {
        words_[word] &= ~mask;
    }
    ones_ += value ? 1 : -1;
    return value ? 1 : -1;
}

template <uint32_t MaxLeafBits> void leaf_bv_node<MaxLeafBits>::insert(uint32_t i, bool value) {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(size_ < max_bits);

    const uint32_t n = ++size_;
    const uint32_t ib = i / word_bits;
    const uint32_t off = i % word_bits;
    const uint32_t nb = (n - 1) / word_bits;

    for (uint32_t b = nb; b > ib; --b) {
        words_[b] = (words_[b] << 1) | (words_[b - 1] >> (word_bits - 1));
    }

    const uint64_t x = words_[ib];
    words_[ib] = (x & ((uint64_t{1} << off) - 1)) | (uint64_t{value} << off);
    if (off < word_bits - 1) {
        words_[ib] |= (x << 1) & (~uint64_t{0} << (off + 1));
    }
    ones_ += value ? 1 : 0;
}

template <uint32_t MaxLeafBits> int32_t leaf_bv_node<MaxLeafBits>::erase(uint32_t i) {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t ib = i / word_bits;
    const uint32_t off = i % word_bits;
    const uint32_t nb = (size_ - 1) / word_bits;
    const bool v = (words_[ib] >> off) & 1;

    words_[ib] = (words_[ib] & ((uint64_t{1} << off) - 1)) | ((words_[ib] >> 1) & (~uint64_t{0} << off));

    for (uint32_t b = ib + 1; b <= nb; ++b) {
        words_[b - 1] |= words_[b] << (word_bits - 1);
        words_[b] >>= 1;
    }

    --size_;
    ones_ -= v ? 1 : 0;
    return v ? -1 : 0;
}

template <uint32_t MaxLeafBits> uint32_t leaf_bv_node<MaxLeafBits>::word_count() const {
    return (size_ + word_bits - 1) / word_bits;
}

} // namespace adaptive
