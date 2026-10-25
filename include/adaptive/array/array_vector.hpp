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
#include <istream>
#include <ostream>
#include <vector>

#include "adaptive/array/array_node_ptr.hpp"
#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"
#include "adaptive/array/dynamic_array_node.hpp"
#include "adaptive/array/leaf_array_node.hpp"
#include "adaptive/array/static_array_node.hpp"

namespace adaptive {

template <uint32_t TargetArity = 64, uint32_t MaxLeafWords = 128, uint32_t Epsilon = 10> class array_vector {
  private:
    static constexpr uint32_t target_arity = TargetArity;
    static constexpr uint32_t child_capacity = 4 * TargetArity;
    static constexpr uint32_t max_leaf_words = MaxLeafWords;
    static constexpr uint32_t epsilon = Epsilon;

    static_assert(target_arity >= 16);
    static_assert((target_arity & (target_arity - 1)) == 0);
    static_assert(child_capacity > 1);
    static_assert(max_leaf_words > 1);
    static_assert(epsilon > 0);

    using leaf_type = leaf_array_node<max_leaf_words>;
    using static_type = static_array_node;

    class dynamic_type;
    using node_type = array_node_ptr<leaf_type, static_type, dynamic_type>;

    class dynamic_type : public dynamic_array_node<node_type, target_arity> {
      public:
        using dynamic_array_node<node_type, target_arity>::dynamic_array_node;
    };

  public:
    explicit array_vector(uint32_t width = 32);
    array_vector(const uint64_t* values, uint64_t size, uint32_t width);
    array_vector(const uint32_t* values, uint64_t size, uint32_t width);

    array_vector(const array_vector& other);
    array_vector& operator=(const array_vector& other);

    array_vector(array_vector&&) noexcept = default;
    array_vector& operator=(array_vector&&) noexcept = default;

    bool empty() const;
    uint64_t size() const;
    uint32_t width() const;
    uint64_t level() const;
    uint64_t space_bytes() const;

    uint64_t access(uint64_t i);
    uint64_t operator[](uint64_t i);
    void read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index = 0);
    std::pair<uint64_t, uint64_t> next(uint64_t i, uint64_t j, uint64_t c);

    void write(uint64_t i, uint64_t value);
    void insert(uint64_t i, uint64_t value);
    void erase(uint64_t i);

    uint64_t theta() const;
    void set_theta(uint64_t value);

    uint64_t serialize(std::ostream& out) const;
    void load(std::istream& in);

  private:
    uint64_t level_ = 0;
    uint32_t width_ = 0;
    adaptive_policy policy_{};
    node_type root_;

    const adaptive_policy& policy() const;
    static uint64_t level_for_size(uint64_t size, uint32_t width);
    static uint64_t value_mask(uint32_t width);

    template <class Value> static std::vector<uint64_t> pack_values(const Value* values, uint64_t size, uint32_t width);

    void assign_packed_root(const uint64_t* packed, uint64_t size);
    void expand_root_if_needed();
    void collapse_root_if_needed();
};

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
array_vector<TargetArity, MaxLeafWords, Epsilon>::array_vector(uint32_t width)
    : width_(width), policy_{.max_flatten_size = 0, .theta = 1000}, root_(node_type::leaf(width)) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= bit_util::word_bits);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
array_vector<TargetArity, MaxLeafWords, Epsilon>::array_vector(const uint64_t* values, uint64_t size, uint32_t width)
    : level_(level_for_size(size, width)), width_(width), policy_{.max_flatten_size = size / Epsilon, .theta = 1000} {
    ADAPTIVE_ASSUME(values != nullptr || size == 0);
    const std::vector<uint64_t> packed = pack_values(values, size, width);
    assign_packed_root(packed.data(), size);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
array_vector<TargetArity, MaxLeafWords, Epsilon>::array_vector(const uint32_t* values, uint64_t size, uint32_t width)
    : level_(level_for_size(size, width)), width_(width), policy_{.max_flatten_size = size / Epsilon, .theta = 1000} {
    ADAPTIVE_ASSUME(values != nullptr || size == 0);
    const std::vector<uint64_t> packed = pack_values(values, size, width);
    assign_packed_root(packed.data(), size);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
array_vector<TargetArity, MaxLeafWords, Epsilon>::array_vector(const array_vector& other)
    : level_(other.level_), width_(other.width_), policy_(other.policy_) {
    const uint64_t n = other.size();
    if (n == 0) {
        root_ = node_type::leaf(width_);
    } else {
        const uint64_t words = (n * width_ + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> packed(words + 1, 0);
        const_cast<array_vector&>(other).read_packed(0, n, packed.data(), 0);
        assign_packed_root(packed.data(), n);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
array_vector<TargetArity, MaxLeafWords, Epsilon>&
array_vector<TargetArity, MaxLeafWords, Epsilon>::operator=(const array_vector& other) {
    if (this != &other) {
        array_vector tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
bool array_vector<TargetArity, MaxLeafWords, Epsilon>::empty() const {
    return size() == 0;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::size() const {
    return root_.size();
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint32_t array_vector<TargetArity, MaxLeafWords, Epsilon>::width() const {
    return width_;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::level() const {
    return level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::space_bytes() const {
    return sizeof(*this) + root_.space_bytes();
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::access(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    return root_.access(i, width_, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::operator[](uint64_t i) {
    return access(i);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::read_packed(uint64_t i, uint64_t len, uint64_t* out,
                                                                   uint64_t out_index) {
    ADAPTIVE_ASSUME(i <= size());
    ADAPTIVE_ASSUME(len <= size() - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);
    root_.read_packed(i, len, out, out_index, width_);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
std::pair<uint64_t, uint64_t> array_vector<TargetArity, MaxLeafWords, Epsilon>::next(uint64_t i, uint64_t j,
                                                                                     uint64_t c) {
    if (i > j || j >= size()) {
        return {0, j + 1};
    }
    return root_.next(i, j, c, width_, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::write(uint64_t i, uint64_t value) {
    ADAPTIVE_ASSUME(i < size());
    ADAPTIVE_ASSUME((value & ~value_mask(width_)) == 0);
    root_.write(i, value, level_, true, width_);
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::insert(uint64_t i, uint64_t value) {
    ADAPTIVE_ASSUME(i <= size());
    ADAPTIVE_ASSUME((value & ~value_mask(width_)) == 0);
    root_.insert(i, value, level_, true, width_);
    expand_root_if_needed();
    policy_.max_flatten_size = size() / Epsilon;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::erase(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    root_.erase(i, level_, true, width_);
    collapse_root_if_needed();
    policy_.max_flatten_size = size() / Epsilon;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::theta() const {
    return policy_.theta;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::set_theta(uint64_t value) {
    ADAPTIVE_ASSUME(value > 0);
    policy_.theta = value;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
const adaptive_policy& array_vector<TargetArity, MaxLeafWords, Epsilon>::policy() const {
    return policy_;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::level_for_size(uint64_t size, uint32_t width) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= bit_util::word_bits);

    uint64_t level = 0;
    uint64_t bound = leaf_type::max_size(width);
    while (size >= bound) {
        ++level;
        bound *= target_arity;
    }
    return level;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::value_mask(uint32_t width) {
    return width == bit_util::word_bits ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
template <class Value>
std::vector<uint64_t> array_vector<TargetArity, MaxLeafWords, Epsilon>::pack_values(const Value* values, uint64_t size,
                                                                                    uint32_t width) {
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= bit_util::word_bits);
    ADAPTIVE_ASSUME(values != nullptr || size == 0);

    const uint64_t words = (size * width + bit_util::word_bits - 1) / bit_util::word_bits;
    std::vector<uint64_t> packed(words + 1);
    const uint64_t mask = value_mask(width);
    for (uint64_t i = 0; i < size; ++i) {
        const uint64_t value = static_cast<uint64_t>(values[i]);
        ADAPTIVE_ASSUME((value & ~mask) == 0);
        if (width == bit_util::word_bits) {
            packed[i] = value;
            continue;
        }
        const uint64_t bit = i * width;
        const uint64_t word = bit / bit_util::word_bits;
        const uint32_t offset = static_cast<uint32_t>(bit % bit_util::word_bits);
        packed[word] |= value << offset;
        if (offset + width > bit_util::word_bits) {
            packed[word + 1] |= value >> (bit_util::word_bits - offset);
        }
    }
    return packed;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::assign_packed_root(const uint64_t* packed, uint64_t size) {
    ADAPTIVE_ASSUME(packed != nullptr || size == 0);

    if (size < leaf_type::max_size(width_)) {
        root_ = node_type::leaf(packed, 0, static_cast<uint32_t>(size), width_);
    } else {
        root_ = node_type::stat(packed, size, width_);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::expand_root_if_needed() {
    if (root_.size() < dynamic_type::weight_upperbound(level_, width_)) {
        return;
    }

    root_.expand_root(width_);
    ++level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::collapse_root_if_needed() {
    if (!root_.collapse_root()) {
        return;
    }
    --level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
uint64_t array_vector<TargetArity, MaxLeafWords, Epsilon>::serialize(std::ostream& out) const {
    const uint64_t n = size();
    const uint32_t w = width();
    const uint64_t th = theta();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&w), sizeof(w));
    out.write(reinterpret_cast<const char*>(&th), sizeof(th));
    uint64_t written = sizeof(n) + sizeof(w) + sizeof(th);
    if (n > 0) {
        const uint64_t words = (n * w + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> packed(words + 1, 0);
        const_cast<array_vector*>(this)->read_packed(0, n, packed.data(), 0);
        const uint64_t packed_count = packed.size();
        const uint64_t packed_bytes = packed_count * sizeof(uint64_t);
        out.write(reinterpret_cast<const char*>(&packed_count), sizeof(packed_count));
        out.write(reinterpret_cast<const char*>(packed.data()), packed_bytes);
        written += sizeof(packed_count) + packed_bytes;
    }
    return written;
}

template <uint32_t TargetArity, uint32_t MaxLeafWords, uint32_t Epsilon>
void array_vector<TargetArity, MaxLeafWords, Epsilon>::load(std::istream& in) {
    uint64_t n = 0;
    uint32_t w = 0;
    uint64_t th = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&w), sizeof(w));
    in.read(reinterpret_cast<char*>(&th), sizeof(th));
    width_ = w;
    policy_.theta = th;
    policy_.max_flatten_size = n / Epsilon;
    if (n == 0) {
        level_ = 0;
        root_ = node_type::leaf(w);
    } else {
        uint64_t packed_count = 0;
        in.read(reinterpret_cast<char*>(&packed_count), sizeof(packed_count));
        std::vector<uint64_t> packed(packed_count);
        in.read(reinterpret_cast<char*>(packed.data()), packed_count * sizeof(uint64_t));
        level_ = level_for_size(n, w);
        assign_packed_root(packed.data(), n);
    }
}

} // namespace adaptive
