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
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"
#include "adaptive/policy.hpp"

namespace adaptive {

template <class NodePtr, uint32_t TargetArity = 64> class dynamic_bv_node {
  public:
    using node_type = NodePtr;

    static constexpr uint32_t target_arity = TargetArity;
    static constexpr uint32_t child_capacity = 4 * TargetArity;

    static uint64_t weight_upperbound(uint64_t level);
    static uint64_t weight_lowerbound(uint64_t level);

    dynamic_bv_node() = default;
    dynamic_bv_node(const uint64_t* data, uint64_t n, uint64_t ones, uint64_t level, bool is_root);

    dynamic_bv_node(const dynamic_bv_node&) = delete;
    dynamic_bv_node& operator=(const dynamic_bv_node&) = delete;

    dynamic_bv_node(dynamic_bv_node&&) noexcept = default;
    dynamic_bv_node& operator=(dynamic_bv_node&&) noexcept = default;

    uint64_t size() const;
    uint64_t ones() const;
    uint64_t space_bytes() const;

    bool access(uint64_t i, const adaptive_policy& policy);
    void read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit);

    template <bool Bit> uint64_t rank(uint64_t i, const adaptive_policy& policy);
    template <bool Bit> uint64_t select(uint64_t j, const adaptive_policy& policy);
    template <bool Bit> int64_t next(uint64_t i, const adaptive_policy& policy);

    int32_t write(uint64_t i, bool value, uint64_t level);
    void insert(uint64_t i, bool value, uint64_t level);
    int32_t erase(uint64_t i, uint64_t level);

    uint16_t child_count() const;

    void record_access();
    void reset_accesses();
    bool must_flatten(const adaptive_policy& policy) const;

    NodePtr take_child(uint32_t k);
    void add_child(NodePtr&& child);
    void split_into(NodePtr* l, NodePtr* r);
    void append_children_from(dynamic_bv_node* right);

    NodePtr& child(uint32_t k) {
        ADAPTIVE_ASSUME(k < child_count_);
        return children_[k];
    }
    const NodePtr& child(uint32_t k) const {
        ADAPTIVE_ASSUME(k < child_count_);
        return children_[k];
    }
    uint32_t find_by_offset(uint64_t offset) const;
    uint64_t prefix_len(uint32_t i) const;
    template <bool Bit> uint32_t find_by_cnt(uint64_t target) const;
    template <bool Bit> uint64_t prefix_cnt(uint32_t i) const;

  private:
    uint16_t child_count_ = 0;
    uint64_t size_ = 0;
    uint64_t ones_ = 0;
    uint64_t accesses_ = 0;

    std::array<NodePtr, child_capacity> children_;
    std::array<uint64_t, child_capacity> child_size_{};
    std::array<uint64_t, child_capacity> child_ones_{};

    template <auto Prefix> uint32_t find_by(uint64_t q) const;
    template <auto Prefix, uint32_t Ell = child_capacity> uint32_t templated_cmov(uint32_t offset, uint64_t q) const;
    static uint64_t prefix_before(const std::array<uint64_t, child_capacity>& p, uint32_t i);
    bool node_overflow(uint32_t k, uint64_t level) const;
    bool node_underflow(uint32_t k, uint64_t level) const;
    void split_child(uint32_t k);
    bool merge_children(uint32_t l, uint32_t r, uint64_t level);

    void recompute_metadata();
};

template <class NodePtr, uint32_t TargetArity>
dynamic_bv_node<NodePtr, TargetArity>::dynamic_bv_node(const uint64_t* data, uint64_t n, uint64_t ones, uint64_t level,
                                                       bool is_root)
    : size_(n), ones_(ones) {
    ADAPTIVE_ASSUME(level != 0);

    const uint64_t total_words = (n + bit_util::word_bits - 1) / bit_util::word_bits;
    const uint64_t child_level = level - 1;
    const uint64_t child_bound = weight_upperbound(child_level);
    uint64_t target_arity = TargetArity;
    if (is_root) {
        const double desired_arity = static_cast<double>(n) / (0.4 * static_cast<double>(child_bound));
        if (desired_arity >= 2.0) {
            target_arity = static_cast<uint64_t>(desired_arity);
        }
    }
    target_arity = std::min<uint64_t>(target_arity, child_capacity);

    const uint64_t base = total_words / target_arity;
    const uint64_t extra = total_words % target_arity;
    uint64_t words_prefix = 0;
    for (uint64_t k = 0; k < target_arity; ++k) {
        const uint64_t words_k = base + (k < extra);
        const uint64_t l = bit_util::word_bits * words_prefix;
        const uint64_t r = k == target_arity - 1 ? n : l + bit_util::word_bits * words_k;
        const uint64_t len = r - l;
        ADAPTIVE_ASSUME(len < child_bound);
        ADAPTIVE_ASSUME(len >= child_bound / 4);
        if (len < NodePtr::max_leaf_bits()) {
            children_[k] = NodePtr::leaf(data + l / bit_util::word_bits, static_cast<uint32_t>(len));
        } else {
            children_[k] = NodePtr::stat(data + l / bit_util::word_bits, len);
        }
        ++child_count_;
        words_prefix += words_k;
    }
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity> uint64_t dynamic_bv_node<NodePtr, TargetArity>::size() const {
    return size_;
}

template <class NodePtr, uint32_t TargetArity> uint64_t dynamic_bv_node<NodePtr, TargetArity>::ones() const {
    return ones_;
}

template <class NodePtr, uint32_t TargetArity> uint64_t dynamic_bv_node<NodePtr, TargetArity>::space_bytes() const {
    uint64_t s = sizeof(*this);
    for (uint32_t k = 0; k < child_count_; ++k) {
        s += children_[k].space_bytes();
    }
    return s;
}

template <class NodePtr, uint32_t TargetArity> uint16_t dynamic_bv_node<NodePtr, TargetArity>::child_count() const {
    return child_count_;
}

template <class NodePtr, uint32_t TargetArity> void dynamic_bv_node<NodePtr, TargetArity>::record_access() {
    ++accesses_;
}

template <class NodePtr, uint32_t TargetArity> void dynamic_bv_node<NodePtr, TargetArity>::reset_accesses() {
    accesses_ = 0;
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_bv_node<NodePtr, TargetArity>::must_flatten(const adaptive_policy& policy) const {
    return size_ <= policy.max_flatten_size && accesses_ * policy.theta >= size_;
}

template <class NodePtr, uint32_t TargetArity>
ADAPTIVE_ALWAYS_INLINE bool dynamic_bv_node<NodePtr, TargetArity>::access(uint64_t i, const adaptive_policy& policy) {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    return children_[k].access(i - l_len, policy);
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_bv_node<NodePtr, TargetArity>::read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit) {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(len <= size_ - i);
    ADAPTIVE_ASSUME(out != nullptr || len == 0);

    uint64_t copied = 0;
    uint64_t prev_len = 0;
    for (uint32_t k = 0; k < child_count_ && copied < len; ++k) {
        const uint64_t r_len = prefix_len(k + 1);
        const uint64_t child_len = r_len - prev_len;
        if (i < r_len) {
            const uint64_t child_i = i > prev_len ? i - prev_len : 0;
            const uint64_t take = std::min(len - copied, child_len - child_i);
            children_[k].read(child_i, take, out, out_bit + copied);
            copied += take;
            i += take;
        }
        prev_len = r_len;
    }
    ADAPTIVE_ASSUME(copied == len);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Bit>
ADAPTIVE_ALWAYS_INLINE uint64_t dynamic_bv_node<NodePtr, TargetArity>::rank(uint64_t i, const adaptive_policy& policy) {
    ADAPTIVE_ASSUME(i <= size_);

    if constexpr (!Bit) {
        return i - rank<1>(i, policy);
    }

    if (i == 0) {
        return 0;
    }
    if (i == size_) {
        return ones_;
    }
    const uint32_t k = find_by_offset(i - 1);
    const uint64_t l_len = prefix_len(k);
    const uint64_t l_ones = prefix_cnt<1>(k);
    return l_ones + children_[k].template rank<1>(i - l_len, policy);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Bit>
ADAPTIVE_ALWAYS_INLINE uint64_t dynamic_bv_node<NodePtr, TargetArity>::select(uint64_t j,
                                                                              const adaptive_policy& policy) {
    uint64_t total;
    if constexpr (Bit) {
        total = ones_;
    } else {
        total = size_ - ones_;
    }

    ADAPTIVE_ASSUME(j > 0);
    ADAPTIVE_ASSUME(j <= total);

    const uint32_t k = find_by_cnt<Bit>(j);
    const uint64_t l_len = prefix_len(k);
    const uint64_t l_cnt = prefix_cnt<Bit>(k);
    return l_len + children_[k].template select<Bit>(j - l_cnt, policy);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Bit>
ADAPTIVE_ALWAYS_INLINE int64_t dynamic_bv_node<NodePtr, TargetArity>::next(uint64_t i, const adaptive_policy& policy) {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t first = find_by_offset(i);
    uint64_t prev_len = prefix_len(first);
    uint64_t prev_cnt = prefix_cnt<Bit>(first);
    for (uint32_t k = first; k < child_count_; ++k) {
        const uint64_t len = prefix_len(k + 1);
        const uint64_t cnt = prefix_cnt<Bit>(k + 1);
        if (cnt == prev_cnt) {
            prev_len = len;
            prev_cnt = cnt;
            continue;
        }
        const uint64_t child_i = k == first ? i - prev_len : 0;
        int64_t next = children_[k].template next<Bit>(child_i, policy);
        if (next != -1) {
            return static_cast<int64_t>(prev_len) + next;
        }
        prev_len = len;
        prev_cnt = cnt;
    }
    return -1;
}

template <class NodePtr, uint32_t TargetArity>
int32_t dynamic_bv_node<NodePtr, TargetArity>::write(uint64_t i, bool value, uint64_t level) {
    ADAPTIVE_ASSUME(i < size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    const uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    const int32_t dif = children_[k].write(i - l_len, value, level - 1, false);
    ones_ += dif;
    for (uint32_t j = k; j < child_count_; ++j) {
        child_ones_[j] += dif;
    }
    return dif;
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_bv_node<NodePtr, TargetArity>::insert(uint64_t i, bool value, uint64_t level) {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    const uint32_t k = find_by_offset(i == size_ ? i - 1 : i);
    const uint64_t l_len = prefix_len(k);
    children_[k].insert(i - l_len, value, level - 1, false);
    ++size_;
    ones_ += value;
    for (uint32_t j = k; j < child_count_; ++j) {
        child_size_[j]++;
        child_ones_[j] += value;
    }
    if (node_overflow(k, level - 1)) {
        split_child(k);
    }
}

template <class NodePtr, uint32_t TargetArity>
int32_t dynamic_bv_node<NodePtr, TargetArity>::erase(uint64_t i, uint64_t level) {
    ADAPTIVE_ASSUME(i < size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    const int32_t dif = children_[k].erase(i - l_len, level - 1, false);
    --size_;
    ones_ += dif;
    for (uint32_t j = k; j < child_count_; ++j) {
        child_size_[j]--;
        child_ones_[j] += dif;
    }
    if (node_underflow(k, level - 1)) {
        ADAPTIVE_ASSUME(child_count_ > 1);
        uint32_t j;
        if (k == 0) {
            j = 0;
            k = 1;
        } else {
            j = k - 1;
        }
        if (merge_children(j, k, level - 1)) {
            std::move(children_.begin() + k + 1, children_.begin() + child_count_, children_.begin() + k);
            --child_count_;
        }
        recompute_metadata();
        if (node_overflow(j, level - 1)) {
            split_child(j);
        }
    }
    return dif;
}

template <class NodePtr, uint32_t TargetArity> NodePtr dynamic_bv_node<NodePtr, TargetArity>::take_child(uint32_t k) {
    ADAPTIVE_ASSUME(k < child_count_);
    return std::move(children_[k]);
}

template <class NodePtr, uint32_t TargetArity> void dynamic_bv_node<NodePtr, TargetArity>::add_child(NodePtr&& child) {
    ADAPTIVE_ASSUME(child_count_ < child_capacity);
    children_[child_count_++] = std::move(child);
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_bv_node<NodePtr, TargetArity>::split_into(NodePtr* l, NodePtr* r) {
    uint32_t k = child_count_;
    for (uint32_t i = 0; i < child_count_; ++i) {
        const uint64_t len = prefix_len(i + 1);
        if (len > size_ - len) {
            k = i;
            break;
        }
    }
    if (k == child_count_) {
        k = child_count_ - 1;
    }
    if (k == 0) {
        k = 1;
    }

    auto left_node = NodePtr::dynamic();
    auto right_node = NodePtr::dynamic();
    auto& left = left_node.as_dynamic();
    auto& right = right_node.as_dynamic();
    std::move(children_.begin(), children_.begin() + k, left.children_.begin());
    std::move(children_.begin() + k, children_.begin() + child_count_, right.children_.begin());
    left.child_count_ = k;
    right.child_count_ = child_count_ - k;
    left.recompute_metadata();
    right.recompute_metadata();
    *l = std::move(left_node);
    *r = std::move(right_node);
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_bv_node<NodePtr, TargetArity>::append_children_from(dynamic_bv_node* right) {
    ADAPTIVE_ASSUME(static_cast<uint32_t>(child_count_) + right->child_count_ <= child_capacity);
    std::move(right->children_.begin(), right->children_.begin() + right->child_count_,
              children_.begin() + child_count_);
    child_count_ += right->child_count_;
    right->child_count_ = 0;
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_bv_node<NodePtr, TargetArity>::find_by_offset(uint64_t offset) const {
    return find_by<&dynamic_bv_node::prefix_len>(offset + 1);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Bit>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_bv_node<NodePtr, TargetArity>::find_by_cnt(uint64_t target) const {
    if constexpr (Bit) {
        return find_by<&dynamic_bv_node::prefix_cnt<1>>(target);
    } else {
        return find_by<&dynamic_bv_node::prefix_cnt<0>>(target);
    }
}

template <class NodePtr, uint32_t TargetArity>
template <auto Prefix>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_bv_node<NodePtr, TargetArity>::find_by(uint64_t q) const {
    uint32_t offset = 0;
    if constexpr (child_capacity > 128) {
        if (child_count_ > 128) {
            offset += ((this->*Prefix)(offset + 128) < q) * 128;
        }
    }
    if constexpr (child_capacity > 64) {
        if (child_count_ > 64) {
            offset += ((this->*Prefix)(offset + 64) < q) * 64;
        }
    }
    return offset + templated_cmov<Prefix, std::min<uint32_t>(child_capacity, 64)>(offset, q);
}

template <class NodePtr, uint32_t TargetArity>
template <auto Prefix, uint32_t Ell>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_bv_node<NodePtr, TargetArity>::templated_cmov(uint32_t offset,
                                                                                      uint64_t q) const {
    static_assert((Ell & (Ell - 1)) == 0);
    static_assert(Ell >= 2);

    if constexpr (Ell == 2) {
        return (this->*Prefix)(offset + 1) < q;
    } else {
        constexpr uint32_t half = Ell / 2;
        const uint32_t step = ((this->*Prefix)(offset + half) < q) * half;
        return step + templated_cmov<Prefix, half>(offset + step, q);
    }
}

template <class NodePtr, uint32_t TargetArity>
ADAPTIVE_ALWAYS_INLINE uint64_t
dynamic_bv_node<NodePtr, TargetArity>::prefix_before(const std::array<uint64_t, child_capacity>& p, uint32_t i) {
    return i == 0 ? 0 : p[i - 1];
}

template <class NodePtr, uint32_t TargetArity>
ADAPTIVE_ALWAYS_INLINE uint64_t dynamic_bv_node<NodePtr, TargetArity>::prefix_len(uint32_t i) const {
    return prefix_before(child_size_, i);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Bit>
ADAPTIVE_ALWAYS_INLINE uint64_t dynamic_bv_node<NodePtr, TargetArity>::prefix_cnt(uint32_t i) const {
    if constexpr (Bit) {
        return prefix_before(child_ones_, i);
    } else {
        return prefix_len(i) - prefix_before(child_ones_, i);
    }
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_bv_node<NodePtr, TargetArity>::weight_upperbound(uint64_t level) {
    uint64_t bound = NodePtr::max_leaf_bits();
    for (uint64_t i = 0; i < level; ++i) {
        bound *= TargetArity;
    }
    return bound;
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_bv_node<NodePtr, TargetArity>::weight_lowerbound(uint64_t level) {
    return weight_upperbound(level) / 4;
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_bv_node<NodePtr, TargetArity>::node_overflow(uint32_t k, uint64_t level) const {
    return children_[k].size() >= weight_upperbound(level);
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_bv_node<NodePtr, TargetArity>::node_underflow(uint32_t k, uint64_t level) const {
    return children_[k].size() < weight_lowerbound(level);
}

template <class NodePtr, uint32_t TargetArity> void dynamic_bv_node<NodePtr, TargetArity>::split_child(uint32_t k) {
    NodePtr l;
    NodePtr r;
    children_[k].split_into(&l, &r);

    ADAPTIVE_ASSUME(static_cast<uint32_t>(child_count_) + 1 <= child_capacity);
    std::move_backward(children_.begin() + k + 1, children_.begin() + child_count_,
                       children_.begin() + child_count_ + 1);
    children_[k] = std::move(l);
    children_[k + 1] = std::move(r);
    ++child_count_;
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_bv_node<NodePtr, TargetArity>::merge_children(uint32_t l, uint32_t r, uint64_t level) {
    return children_[l].merge_with(&children_[r], level);
}

template <class NodePtr, uint32_t TargetArity> void dynamic_bv_node<NodePtr, TargetArity>::recompute_metadata() {
    uint64_t size_prefix = 0;
    uint64_t ones_prefix = 0;
    for (uint32_t k = 0; k < child_count_; ++k) {
        size_prefix += children_[k].size();
        ones_prefix += children_[k].ones();
        child_size_[k] = size_prefix;
        child_ones_[k] = ones_prefix;
    }
    for (uint32_t k = child_count_; k < child_capacity; ++k) {
        children_[k].reset();
        child_size_[k] = std::numeric_limits<uint64_t>::max();
        child_ones_[k] = std::numeric_limits<uint64_t>::max() >> 1;
    }
    size_ = size_prefix;
    ones_ = ones_prefix;
}

} // namespace adaptive
