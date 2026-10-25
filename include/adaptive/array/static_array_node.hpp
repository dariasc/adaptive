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
#include <utility>
#include <vector>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"

namespace adaptive {

class static_array_node {
  public:
    static constexpr uint32_t word_bits = 64;

    static_array_node(const uint64_t* packed, uint64_t size, uint32_t width);
    static_array_node(const uint64_t* packed, uint64_t offset, uint64_t size, uint32_t width);
    static_array_node(std::vector<uint64_t>&& words, uint64_t size, uint32_t width);

    uint64_t size() const;
    uint32_t width() const;
    uint64_t space_bytes() const;
    const uint64_t* data() const;

    uint64_t access(uint64_t i) const;
    uint64_t last() const;
    std::pair<uint64_t, uint64_t> next(uint64_t i, uint64_t j, uint64_t c) const;
    void read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index) const;

  private:
    uint64_t size_ = 0;
    uint8_t width_ = 0;
    std::vector<uint64_t> words_;

    static uint64_t mask(uint32_t width);
};

inline static_array_node::static_array_node(const uint64_t* packed, uint64_t size, uint32_t width)
    : static_array_node(packed, 0, size, width) {}

inline static_array_node::static_array_node(const uint64_t* packed, uint64_t offset, uint64_t size, uint32_t width)
    : size_(size), width_(static_cast<uint8_t>(width)) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= word_bits);
    ADAPTIVE_ASSUME(size > 0);
    ADAPTIVE_ASSUME(packed != nullptr);

    const uint64_t words = (static_cast<uint64_t>(size_) * width_ + word_bits - 1) / word_bits;
    words_.resize(words);
    bit_util::copy_bits(words_.data(), 0, packed, offset * width, size * width);
}

inline static_array_node::static_array_node(std::vector<uint64_t>&& words, uint64_t size, uint32_t width)
    : size_(size), width_(static_cast<uint8_t>(width)), words_(std::move(words)) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= word_bits);
    ADAPTIVE_ASSUME(size > 0);

    const uint64_t expected_words = (static_cast<uint64_t>(size_) * width_ + word_bits - 1) / word_bits;
    words_.resize(expected_words);
    const uint64_t total_bits = size_ * width_;
    if (total_bits % word_bits != 0) {
        words_.back() &= (uint64_t{1} << (total_bits % word_bits)) - 1;
    }
}

inline uint64_t static_array_node::size() const {
    return size_;
}

inline uint32_t static_array_node::width() const {
    return width_;
}

inline uint64_t static_array_node::space_bytes() const {
    return sizeof(*this) + words_.capacity() * sizeof(uint64_t);
}

inline const uint64_t* static_array_node::data() const {
    return words_.data();
}

inline uint64_t static_array_node::access(uint64_t i) const {
    ADAPTIVE_ASSUME(i < size_);

    if (width_ == word_bits) {
        return words_[i];
    }
    const uint64_t bit = i * width_;
    const uint64_t word = bit / word_bits;
    const uint32_t offset = static_cast<uint32_t>(bit % word_bits);
    uint64_t value = words_[word] >> offset;
    if (offset + width_ > word_bits) {
        value |= words_[word + 1] << (word_bits - offset);
    }
    return value & mask(width_);
}

inline uint64_t static_array_node::last() const {
    ADAPTIVE_ASSUME(size_ > 0);
    return access(size_ - 1);
}

inline void static_array_node::read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index) const {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(len <= size_ - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);

    bit_util::copy_bits(out, out_index * width_, words_.data(), i * width_, len * width_);
}

inline uint64_t static_array_node::mask(uint32_t width) {
    return width == word_bits ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
}

inline std::pair<uint64_t, uint64_t> static_array_node::next(uint64_t i, uint64_t j, uint64_t c) const {
    ADAPTIVE_ASSUME(i <= j);
    ADAPTIVE_ASSUME(j < size_);

    const uint64_t vi = access(i);
    if (vi >= c)
        return {vi, i};
    if (access(j) < c)
        return {0, j + 1};

    uint64_t d = 1;
    while (i + d <= j && access(i + d) < c) {
        i += d;
        d <<= 1;
    }
    uint64_t low = i + 1, high = std::min(j, i + d);
    while (low < high) {
        const uint64_t mid = low + (high - low) / 2;
        if (access(mid) < c)
            low = mid + 1;
        else
            high = mid;
    }
    return {access(low), low};
}

} // namespace adaptive
