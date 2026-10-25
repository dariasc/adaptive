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
#include <utility>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"

namespace adaptive {

template <uint32_t MaxLeafWords = 128> class leaf_array_node {
  public:
    static constexpr uint32_t word_bits = 64;
    static constexpr uint32_t max_leaf_words = MaxLeafWords;

    static_assert(max_leaf_words > 1);

    explicit leaf_array_node(uint32_t width);
    leaf_array_node(const uint64_t* packed, uint64_t offset, uint32_t size, uint32_t width);

    uint32_t size() const;
    uint32_t width() const;
    uint64_t space_bytes() const;
    const uint64_t* data() const;

    uint64_t access(uint32_t i) const;
    uint64_t last() const;
    std::pair<uint64_t, uint32_t> next(uint32_t i, uint32_t j, uint64_t c) const;
    void read_packed(uint32_t i, uint32_t len, uint64_t* out, uint64_t out_index) const;

    void write(uint32_t i, uint64_t value);
    void insert(uint32_t i, uint64_t value);
    void erase(uint32_t i);

    static uint32_t max_size(uint32_t width);

  private:
    uint32_t size_ = 0;
    uint8_t width_ = 0;
    std::array<uint64_t, max_leaf_words + 1> words_{};

    static uint64_t mask(uint32_t width);
    static uint64_t packed_access(const uint64_t* words, uint64_t i, uint32_t width);
    static void packed_write(uint64_t* words, uint64_t i, uint32_t width, uint64_t value);
};

template <uint32_t MaxLeafWords>
leaf_array_node<MaxLeafWords>::leaf_array_node(uint32_t width) : width_(static_cast<uint8_t>(width)) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= word_bits);
}

template <uint32_t MaxLeafWords>
leaf_array_node<MaxLeafWords>::leaf_array_node(const uint64_t* packed, uint64_t offset, uint32_t size, uint32_t width)
    : size_(size), width_(static_cast<uint8_t>(width)) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= word_bits);
    ADAPTIVE_ASSUME(size <= max_size(width));
    ADAPTIVE_ASSUME(packed != nullptr || size == 0);

    if (size != 0) {
        bit_util::copy_bits(words_.data(), 0, packed, offset * width, static_cast<uint64_t>(size) * width);
    }
}

template <uint32_t MaxLeafWords> uint32_t leaf_array_node<MaxLeafWords>::size() const {
    return size_;
}

template <uint32_t MaxLeafWords> uint32_t leaf_array_node<MaxLeafWords>::width() const {
    return width_;
}

template <uint32_t MaxLeafWords> uint64_t leaf_array_node<MaxLeafWords>::space_bytes() const {
    return sizeof(*this);
}

template <uint32_t MaxLeafWords> const uint64_t* leaf_array_node<MaxLeafWords>::data() const {
    return words_.data();
}

template <uint32_t MaxLeafWords> uint64_t leaf_array_node<MaxLeafWords>::access(uint32_t i) const {
    ADAPTIVE_ASSUME(i < size_);
    return packed_access(words_.data(), i, width_);
}

template <uint32_t MaxLeafWords> uint64_t leaf_array_node<MaxLeafWords>::last() const {
    ADAPTIVE_ASSUME(size_ > 0);
    return packed_access(words_.data(), size_ - 1, width_);
}

template <uint32_t MaxLeafWords>
void leaf_array_node<MaxLeafWords>::read_packed(uint32_t i, uint32_t len, uint64_t* out, uint64_t out_index) const {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(len <= size_ - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);

    bit_util::copy_bits(out, out_index * width_, words_.data(), static_cast<uint64_t>(i) * width_,
                        static_cast<uint64_t>(len) * width_);
}

template <uint32_t MaxLeafWords> void leaf_array_node<MaxLeafWords>::write(uint32_t i, uint64_t value) {
    ADAPTIVE_ASSUME(i < size_);
    ADAPTIVE_ASSUME((value & ~mask(width_)) == 0);
    packed_write(words_.data(), i, width_, value);
}

template <uint32_t MaxLeafWords> void leaf_array_node<MaxLeafWords>::insert(uint32_t i, uint64_t value) {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(size_ < max_size(width_));
    ADAPTIVE_ASSUME((value & ~mask(width_)) == 0);

    const uint64_t old_bits = static_cast<uint64_t>(size_) * width_;
    const uint32_t first_word = static_cast<uint32_t>((static_cast<uint64_t>(i) * width_) / word_bits);
    const uint32_t offset = static_cast<uint32_t>((static_cast<uint64_t>(i) * width_) % word_bits);
    const uint32_t last_word = static_cast<uint32_t>((old_bits + word_bits - 1) / word_bits);
    ++size_;

    if (width_ == word_bits) {
        for (uint32_t word = last_word; word > first_word; --word) {
            words_[word] = words_[word - 1];
        }
        words_[first_word] = value;
        return;
    }

    for (uint32_t word = last_word; word > first_word; --word) {
        words_[word] = (words_[word] << width_) | (words_[word - 1] >> (word_bits - width_));
    }

    const uint64_t low_mask = (uint64_t{1} << offset) - 1;
    const uint64_t old = words_[first_word];
    words_[first_word] = (old & low_mask) | (value << offset);
    if (offset + width_ < word_bits) {
        words_[first_word] |= (old << width_) & (~uint64_t{0} << (offset + width_));
    } else {
        words_[first_word + 1] = (words_[first_word + 1] & (~uint64_t{0} << (offset + width_ - word_bits))) |
                                 (value >> (word_bits - offset));
    }
}

template <uint32_t MaxLeafWords> void leaf_array_node<MaxLeafWords>::erase(uint32_t i) {
    ADAPTIVE_ASSUME(i < size_);

    const uint64_t old_bits = static_cast<uint64_t>(size_) * width_;
    const uint32_t first_word = static_cast<uint32_t>((static_cast<uint64_t>(i) * width_) / word_bits);
    const uint32_t offset = static_cast<uint32_t>((static_cast<uint64_t>(i) * width_) % word_bits);
    const uint32_t last_word = static_cast<uint32_t>((old_bits - 1) / word_bits);
    --size_;

    if (width_ == word_bits) {
        for (uint32_t word = first_word; word < last_word; ++word) {
            words_[word] = words_[word + 1];
        }
        words_[last_word] = 0;
        return;
    }

    const uint64_t low_mask = (uint64_t{1} << offset) - 1;
    const uint64_t low = words_[first_word] & low_mask;
    for (uint32_t word = first_word; word < last_word; ++word) {
        words_[word] = (words_[word] >> width_) | (words_[word + 1] << (word_bits - width_));
    }
    words_[last_word] >>= width_;
    words_[first_word] = low | (words_[first_word] & ~low_mask);

    const uint64_t bits = static_cast<uint64_t>(size_) * width_;
    const uint32_t used_words = static_cast<uint32_t>((bits + word_bits - 1) / word_bits);
    if (used_words == 0) {
        words_[0] = 0;
        return;
    }
    if (bits % word_bits != 0) {
        words_[used_words - 1] &= (uint64_t{1} << (bits % word_bits)) - 1;
    }
    words_[used_words] = 0;
}

template <uint32_t MaxLeafWords> uint32_t leaf_array_node<MaxLeafWords>::max_size(uint32_t width) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= word_bits);
    return ((max_leaf_words * word_bits / width) / 2) * 2;
}

template <uint32_t MaxLeafWords> uint64_t leaf_array_node<MaxLeafWords>::mask(uint32_t width) {
    return width == word_bits ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
}

template <uint32_t MaxLeafWords>
uint64_t leaf_array_node<MaxLeafWords>::packed_access(const uint64_t* words, uint64_t i, uint32_t width) {
    if (width == word_bits) {
        return words[i];
    }
    const uint64_t bit = i * width;
    const uint64_t word = bit / word_bits;
    const uint32_t offset = static_cast<uint32_t>(bit % word_bits);
    uint64_t value = words[word] >> offset;
    if (offset + width > word_bits) {
        value |= words[word + 1] << (word_bits - offset);
    }
    return value & mask(width);
}

template <uint32_t MaxLeafWords>
void leaf_array_node<MaxLeafWords>::packed_write(uint64_t* words, uint64_t i, uint32_t width, uint64_t value) {
    if (width == word_bits) {
        words[i] = value;
        return;
    }
    const uint64_t bit = i * width;
    const uint64_t word = bit / word_bits;
    const uint32_t offset = static_cast<uint32_t>(bit % word_bits);
    const uint64_t value_mask = mask(width);
    words[word] &= ~(value_mask << offset);
    words[word] |= value << offset;
    if (offset + width > word_bits) {
        words[word + 1] &= ~(value_mask >> (word_bits - offset));
        words[word + 1] |= value >> (word_bits - offset);
    }
}

template <uint32_t MaxLeafWords>
std::pair<uint64_t, uint32_t> leaf_array_node<MaxLeafWords>::next(uint32_t i, uint32_t j, uint64_t c) const {
    ADAPTIVE_ASSUME(i <= j);
    ADAPTIVE_ASSUME(j < size_);

    const uint64_t vi = access(i);
    if (vi >= c)
        return {vi, i};
    if (access(j) < c)
        return {0, j + 1};

    uint32_t d = 1;
    while (i + d <= j && access(i + d) < c) {
        i += d;
        d <<= 1;
    }
    uint32_t low = i + 1, high = std::min(j, i + d);
    while (low < high) {
        const uint32_t mid = low + (high - low) / 2;
        if (access(mid) < c)
            low = mid + 1;
        else
            high = mid;
    }
    return {access(low), low};
}

} // namespace adaptive
