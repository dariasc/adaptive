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

#include "adaptive/assert.hpp"
#include "adaptive/bit_vector/dynamic_bv_node.hpp"
#include "adaptive/bit_vector/leaf_bv_node.hpp"
#include "adaptive/bit_vector/bv_node_ptr.hpp"
#include "adaptive/bit_vector/static_bv_node.hpp"

namespace adaptive {

template <uint32_t TargetArity = 64, uint32_t MaxLeafBits = 8192, uint32_t Epsilon = 10> class bit_vector {
  private:
    static constexpr uint32_t target_arity = TargetArity;
    static constexpr uint32_t child_capacity = 4 * TargetArity;
    static constexpr uint32_t max_leaf_bits = MaxLeafBits;
    static constexpr uint32_t epsilon = Epsilon;

    static_assert(target_arity >= 16);
    static_assert((target_arity & (target_arity - 1)) == 0);
    static_assert(child_capacity > 1);
    static_assert(max_leaf_bits > 0);
    static_assert(epsilon > 0);

    using leaf_type = leaf_bv_node<max_leaf_bits>;
    using static_type = static_bv_node<4>;

    class dynamic_type;
    using node_type = bv_node_ptr<leaf_type, static_type, dynamic_type>;

    class dynamic_type : public dynamic_bv_node<node_type, target_arity> {
      public:
        using dynamic_bv_node<node_type, target_arity>::dynamic_bv_node;
    };

  public:
    bit_vector();
    bit_vector(const uint64_t* data, uint64_t size);

    bit_vector(const bit_vector& other);
    bit_vector& operator=(const bit_vector& other);

    bit_vector(bit_vector&&) noexcept = default;
    bit_vector& operator=(bit_vector&&) noexcept = default;

    bool empty() const;
    uint64_t size() const;
    uint64_t ones() const;
    uint64_t level() const;
    uint64_t space_bytes() const;

    bool access(uint64_t i);
    bool operator[](uint64_t i);
    void read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit);

    template <bool Bit> uint64_t rank(uint64_t i);
    uint64_t rank1(uint64_t i);
    uint64_t rank0(uint64_t i);

    template <bool Bit> uint64_t select(uint64_t j);
    uint64_t select1(uint64_t j);
    uint64_t select0(uint64_t j);

    template <bool Bit> int64_t next(uint64_t i);
    int64_t next1(uint64_t i);
    int64_t next0(uint64_t i);

    int32_t write(uint64_t i, bool value);
    void insert(uint64_t i, bool value);
    int32_t erase(uint64_t i);

    uint64_t theta() const;
    void set_theta(uint64_t value);

    uint64_t serialize(std::ostream& out) const;
    void load(std::istream& in);

  private:
    uint64_t level_ = 0;
    adaptive_policy policy_{};
    node_type root_;

    const adaptive_policy& policy() const;
    static uint64_t level_for_size(uint64_t size);

    void expand_root_if_needed();
    void collapse_root_if_needed();
};

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bit_vector<TargetArity, MaxLeafBits, Epsilon>::bit_vector()
    : policy_{.max_flatten_size = 0, .theta = 1000}, root_(node_type::leaf()) {}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bit_vector<TargetArity, MaxLeafBits, Epsilon>::bit_vector(const uint64_t* data, uint64_t size)
    : level_(level_for_size(size)), policy_{.max_flatten_size = size / Epsilon, .theta = 1000} {
    ADAPTIVE_ASSUME(data != nullptr || size == 0);

    if (size < max_leaf_bits) {
        root_ = node_type::leaf(data, static_cast<uint32_t>(size));
    } else {
        root_ = node_type::stat(data, size);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bit_vector<TargetArity, MaxLeafBits, Epsilon>::bit_vector(const bit_vector& other)
    : level_(other.level_), policy_(other.policy_) {
    const uint64_t n = other.size();
    if (n == 0) {
        root_ = node_type::leaf();
    } else {
        const uint64_t words = (n + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> buf(words, 0);
        const_cast<bit_vector&>(other).read(0, n, buf.data(), 0);
        if (n < max_leaf_bits) {
            root_ = node_type::leaf(buf.data(), static_cast<uint32_t>(n));
        } else {
            root_ = node_type::stat(buf.data(), n);
        }
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bit_vector<TargetArity, MaxLeafBits, Epsilon>&
bit_vector<TargetArity, MaxLeafBits, Epsilon>::operator=(const bit_vector& other) {
    if (this != &other) {
        bit_vector tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bool bit_vector<TargetArity, MaxLeafBits, Epsilon>::empty() const {
    return size() == 0;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::size() const {
    return root_.size();
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::ones() const {
    return root_.ones();
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::level() const {
    return level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::space_bytes() const {
    return sizeof(*this) + root_.space_bytes();
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bool bit_vector<TargetArity, MaxLeafBits, Epsilon>::access(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    return root_.access(i, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
bool bit_vector<TargetArity, MaxLeafBits, Epsilon>::operator[](uint64_t i) {
    return access(i);
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit) {
    ADAPTIVE_ASSUME(i <= size());
    ADAPTIVE_ASSUME(len <= size() - i);
    root_.read(i, len, out, out_bit);
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
template <bool Bit>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::rank(uint64_t i) {
    if constexpr (Bit) {
        return rank1(i);
    } else {
        return rank0(i);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::rank1(uint64_t i) {
    ADAPTIVE_ASSUME(i <= size());
    return root_.template rank<1>(i, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::rank0(uint64_t i) {
    return root_.template rank<0>(i, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
template <bool Bit>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::select(uint64_t j) {
    if constexpr (Bit) {
        return select1(j);
    } else {
        return select0(j);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::select1(uint64_t j) {
    ADAPTIVE_ASSUME(j > 0);
    ADAPTIVE_ASSUME(j <= ones());
    return root_.template select<1>(j, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::select0(uint64_t j) {
    ADAPTIVE_ASSUME(j > 0);
    ADAPTIVE_ASSUME(j <= size() - ones());
    return root_.template select<0>(j, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
template <bool Bit>
int64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::next(uint64_t i) {
    if constexpr (Bit) {
        return next1(i);
    } else {
        return next0(i);
    }
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
int64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::next1(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    return root_.template next<1>(i, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
int64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::next0(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    return root_.template next<0>(i, policy());
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
int32_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::write(uint64_t i, bool value) {
    ADAPTIVE_ASSUME(i < size());
    return root_.write(i, value, level_, true);
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::insert(uint64_t i, bool value) {
    ADAPTIVE_ASSUME(i <= size());
    root_.insert(i, value, level_, true);
    expand_root_if_needed();
    policy_.max_flatten_size = size() / Epsilon;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
int32_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::erase(uint64_t i) {
    ADAPTIVE_ASSUME(i < size());
    const int32_t dif = root_.erase(i, level_, true);
    collapse_root_if_needed();
    policy_.max_flatten_size = size() / Epsilon;
    return dif;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::theta() const {
    return policy_.theta;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::set_theta(uint64_t value) {
    ADAPTIVE_ASSUME(value > 0);
    policy_.theta = value;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
const adaptive_policy& bit_vector<TargetArity, MaxLeafBits, Epsilon>::policy() const {
    return policy_;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::level_for_size(uint64_t size) {
    uint64_t level = 0;
    uint64_t bound = max_leaf_bits;
    while (size >= bound) {
        ++level;
        bound *= target_arity;
    }
    return level;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::expand_root_if_needed() {
    if (root_.size() < dynamic_type::weight_upperbound(level_)) {
        return;
    }

    root_.expand_root();
    ++level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::collapse_root_if_needed() {
    if (!root_.collapse_root()) {
        return;
    }
    --level_;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
uint64_t bit_vector<TargetArity, MaxLeafBits, Epsilon>::serialize(std::ostream& out) const {
    const uint64_t n = size();
    const uint64_t th = theta();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&th), sizeof(th));
    uint64_t written = sizeof(n) + sizeof(th);
    if (n > 0) {
        const uint64_t word_count = (n + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> buf(word_count, 0);
        const_cast<bit_vector*>(this)->read(0, n, buf.data(), 0);
        const uint64_t data_bytes = word_count * sizeof(uint64_t);
        out.write(reinterpret_cast<const char*>(buf.data()), data_bytes);
        written += data_bytes;
    }
    return written;
}

template <uint32_t TargetArity, uint32_t MaxLeafBits, uint32_t Epsilon>
void bit_vector<TargetArity, MaxLeafBits, Epsilon>::load(std::istream& in) {
    uint64_t n = 0;
    uint64_t th = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&th), sizeof(th));
    if (n == 0) {
        *this = bit_vector();
    } else {
        const uint64_t word_count = (n + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> buf(word_count, 0);
        in.read(reinterpret_cast<char*>(buf.data()), word_count * sizeof(uint64_t));
        *this = bit_vector(buf.data(), n);
    }
    set_theta(th);
}

} // namespace adaptive
