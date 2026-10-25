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

template <class NodePtr, uint32_t TargetArity = 64> class dynamic_array_node {
  public:
    using node_type = NodePtr;

    static constexpr uint32_t target_arity = TargetArity;
    static constexpr uint32_t child_capacity = 4 * TargetArity;

    static_assert(target_arity > 0);
    static_assert(child_capacity > 1);

    static uint64_t weight_upperbound(uint64_t level, uint32_t width);
    static uint64_t weight_lowerbound(uint64_t level, uint32_t width);

    dynamic_array_node() = default;
    dynamic_array_node(const uint64_t* packed, uint64_t n, uint64_t level, bool is_root, uint32_t width);

    dynamic_array_node(const dynamic_array_node&) = delete;
    dynamic_array_node& operator=(const dynamic_array_node&) = delete;

    dynamic_array_node(dynamic_array_node&&) noexcept = default;
    dynamic_array_node& operator=(dynamic_array_node&&) noexcept = default;

    uint64_t size() const;
    uint64_t space_bytes() const;

    uint64_t access(uint64_t i, uint32_t width, const adaptive_policy& policy);
    std::pair<uint64_t, uint64_t> next(uint64_t i, uint64_t j, uint64_t c, uint32_t width,
                                       const adaptive_policy& policy);
    void read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index, uint32_t width);

    void write(uint64_t i, uint64_t value, uint64_t level, uint32_t width);
    void insert(uint64_t i, uint64_t value, uint64_t level, uint32_t width);
    void erase(uint64_t i, uint64_t level, uint32_t width);

    uint16_t child_count() const;

    void record_access();
    void reset_accesses();
    bool must_flatten(const adaptive_policy& policy) const;

    NodePtr take_child(uint32_t k);
    void add_child(NodePtr&& child);
    void split_into(NodePtr* l, NodePtr* r);
    void append_children_from(dynamic_array_node* right);

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
    uint64_t last() const {
        return last_;
    }

  private:
    uint16_t child_count_ = 0;
    uint64_t size_ = 0;
    uint64_t accesses_ = 0;
    uint64_t last_ = 0;

    std::array<NodePtr, child_capacity> children_;
    std::array<uint64_t, child_capacity> child_size_{};

    uint32_t find_for_insert(uint64_t offset) const;

    template <bool Interp = true> uint32_t find_by(uint64_t q, uint64_t t) const;
    template <uint32_t Ell = child_capacity> uint32_t templated_cmov(uint32_t offset, uint64_t q) const;
    static uint64_t prefix_before(const std::array<uint64_t, child_capacity>& p, uint32_t i);
    bool node_overflow(uint32_t k, uint64_t level, uint32_t width) const;
    bool node_underflow(uint32_t k, uint64_t level, uint32_t width) const;
    void split_child(uint32_t k, uint32_t width);
    bool merge_children(uint32_t l, uint32_t r, uint64_t level, uint32_t width);

    void recompute_metadata();
};

template <class NodePtr, uint32_t TargetArity>
dynamic_array_node<NodePtr, TargetArity>::dynamic_array_node(const uint64_t* packed, uint64_t n, uint64_t level,
                                                             bool is_root, uint32_t width)
    : size_(n) {
    ADAPTIVE_ASSUME(level != 0);
    ADAPTIVE_ASSUME(width > 0);
    ADAPTIVE_ASSUME(width <= bit_util::word_bits);
    ADAPTIVE_ASSUME(packed != nullptr || n == 0);

    const uint64_t child_level = level - 1;
    const uint64_t child_bound = weight_upperbound(child_level, width);
    uint64_t target_arity = TargetArity;
    if (is_root) {
        const double desired_arity = static_cast<double>(n) / (0.4 * static_cast<double>(child_bound));
        if (desired_arity >= 2.0) {
            target_arity = static_cast<uint64_t>(desired_arity);
        }
    }
    target_arity = std::min<uint64_t>(target_arity, child_capacity);

    const uint64_t base = n / target_arity;
    const uint64_t extra = n % target_arity;
    uint64_t elem_prefix = 0;
    for (uint64_t k = 0; k < target_arity; ++k) {
        const uint64_t elems_k = base + (k < extra);
        const uint64_t l = elem_prefix;
        const uint64_t r = k == target_arity - 1 ? n : l + elems_k;
        const uint64_t len = r - l;
        ADAPTIVE_ASSUME(len < child_bound);
        ADAPTIVE_ASSUME(len >= child_bound / 4);
        if (len < NodePtr::max_leaf_size(width)) {
            children_[k] = NodePtr::leaf(packed, l, static_cast<uint32_t>(len), width);
        } else {
            children_[k] = NodePtr::stat(packed, l, len, width);
        }
        ++child_count_;
        elem_prefix += elems_k;
    }
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity> uint64_t dynamic_array_node<NodePtr, TargetArity>::size() const {
    return size_;
}

template <class NodePtr, uint32_t TargetArity> uint64_t dynamic_array_node<NodePtr, TargetArity>::space_bytes() const {
    uint64_t s = sizeof(*this);
    for (uint32_t k = 0; k < child_count_; ++k) {
        s += children_[k].space_bytes();
    }
    return s;
}

template <class NodePtr, uint32_t TargetArity> uint16_t dynamic_array_node<NodePtr, TargetArity>::child_count() const {
    return child_count_;
}

template <class NodePtr, uint32_t TargetArity> void dynamic_array_node<NodePtr, TargetArity>::record_access() {
    ++accesses_;
}

template <class NodePtr, uint32_t TargetArity> void dynamic_array_node<NodePtr, TargetArity>::reset_accesses() {
    accesses_ = 0;
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_array_node<NodePtr, TargetArity>::must_flatten(const adaptive_policy& policy) const {
    return size_ <= policy.max_flatten_size && accesses_ * policy.theta >= size_;
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_array_node<NodePtr, TargetArity>::access(uint64_t i, uint32_t width, const adaptive_policy& policy) {
    ADAPTIVE_ASSUME(i < size_);

    const uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    return children_[k].access(i - l_len, width, policy);
}

template <class NodePtr, uint32_t TargetArity>
std::pair<uint64_t, uint64_t> dynamic_array_node<NodePtr, TargetArity>::next(uint64_t i, uint64_t j, uint64_t c,
                                                                             uint32_t width,
                                                                             const adaptive_policy& policy) {
    ADAPTIVE_ASSUME(i <= j);
    ADAPTIVE_ASSUME(j < size_);

    if (j == size_ - 1 && last_ < c) {
        return {0, j + 1};
    }

    const uint32_t start_k = find_by_offset(i);
    const uint32_t end_k = find_by_offset(j);

    for (uint32_t k = start_k; k <= end_k; ++k) {
        if (k < end_k && children_[k].last() < c) {
            continue;
        }

        const uint64_t l_len = prefix_len(k);
        const uint64_t child_size = children_[k].size();
        const uint64_t ci = (k == start_k) ? (i - l_len) : 0;
        const uint64_t cj = (k == end_k) ? (j - l_len) : (child_size - 1);

        auto res = children_[k].next(ci, cj, c, width, policy);
        if (res.second <= cj) {
            return {res.first, res.second + l_len};
        }
    }
    return {0, j + 1};
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index,
                                                           uint32_t width) {
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
            children_[k].read_packed(child_i, take, out, out_index + copied, width);
            copied += take;
            i += take;
        }
        prev_len = r_len;
    }
    ADAPTIVE_ASSUME(copied == len);
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::write(uint64_t i, uint64_t value, uint64_t level, uint32_t width) {
    ADAPTIVE_ASSUME(i < size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    const uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    children_[k].write(i - l_len, value, level - 1, false, width);
    if (i == size_ - 1) {
        last_ = value;
    }
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::insert(uint64_t i, uint64_t value, uint64_t level, uint32_t width) {
    ADAPTIVE_ASSUME(i <= size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    const uint32_t k = find_for_insert(i);
    const uint64_t l_len = prefix_len(k);
    children_[k].insert(i - l_len, value, level - 1, false, width);
    ++size_;
    for (uint32_t j = k; j < child_count_; ++j) {
        child_size_[j]++;
    }
    if (node_overflow(k, level - 1, width)) {
        split_child(k, width);
    } else {
        last_ = children_[child_count_ - 1].last();
    }
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::erase(uint64_t i, uint64_t level, uint32_t width) {
    ADAPTIVE_ASSUME(i < size_);
    ADAPTIVE_ASSUME(level > 0);

    reset_accesses();
    uint32_t k = find_by_offset(i);
    const uint64_t l_len = prefix_len(k);
    children_[k].erase(i - l_len, level - 1, false, width);
    --size_;
    for (uint32_t j = k; j < child_count_; ++j) {
        child_size_[j]--;
    }
    if (node_underflow(k, level - 1, width)) {
        ADAPTIVE_ASSUME(child_count_ > 1);
        uint32_t j;
        if (k == 0) {
            j = 0;
            k = 1;
        } else {
            j = k - 1;
        }
        if (merge_children(j, k, level - 1, width)) {
            std::move(children_.begin() + k + 1, children_.begin() + child_count_, children_.begin() + k);
            --child_count_;
        }
        recompute_metadata();
        if (node_overflow(j, level - 1, width)) {
            split_child(j, width);
        }
    } else {
        last_ = (child_count_ > 0 && size_ > 0) ? children_[child_count_ - 1].last() : 0;
    }
}

template <class NodePtr, uint32_t TargetArity>
NodePtr dynamic_array_node<NodePtr, TargetArity>::take_child(uint32_t k) {
    ADAPTIVE_ASSUME(k < child_count_);
    return std::move(children_[k]);
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::add_child(NodePtr&& child) {
    ADAPTIVE_ASSUME(child_count_ < child_capacity);
    children_[child_count_++] = std::move(child);
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::split_into(NodePtr* l, NodePtr* r) {
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
void dynamic_array_node<NodePtr, TargetArity>::append_children_from(dynamic_array_node* right) {
    ADAPTIVE_ASSUME(static_cast<uint32_t>(child_count_) + right->child_count_ <= child_capacity);
    std::move(right->children_.begin(), right->children_.begin() + right->child_count_,
              children_.begin() + child_count_);
    child_count_ += right->child_count_;
    right->child_count_ = 0;
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
uint32_t dynamic_array_node<NodePtr, TargetArity>::find_by_offset(uint64_t offset) const {
    return find_by(offset + 1, size_);
}

template <class NodePtr, uint32_t TargetArity>
uint32_t dynamic_array_node<NodePtr, TargetArity>::find_for_insert(uint64_t offset) const {
    return find_by(offset, size_);
}

template <class NodePtr, uint32_t TargetArity>
template <bool Interp>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_array_node<NodePtr, TargetArity>::find_by(uint64_t q, uint64_t t) const {
    if constexpr (Interp) {
        if (t != 0) {
            const uint32_t i = static_cast<uint32_t>((q * child_count_) / t);
            if ((i < child_count_) && (prefix_len(i + 1) >= q) && ((i == 0) || (prefix_len(i) < q))) {
                return i;
            }
        }
    }
    uint32_t offset = 0;
    if constexpr (child_capacity > 128) {
        if (child_count_ > 128) {
            offset += (prefix_len(offset + 128) < q) * 128;
        }
    }
    if constexpr (child_capacity > 64) {
        if (child_count_ > 64) {
            offset += (prefix_len(offset + 64) < q) * 64;
        }
    }
    return offset + templated_cmov<std::min<uint32_t>(child_capacity, 64)>(offset, q);
}

template <class NodePtr, uint32_t TargetArity>
template <uint32_t Ell>
ADAPTIVE_ALWAYS_INLINE uint32_t dynamic_array_node<NodePtr, TargetArity>::templated_cmov(uint32_t offset,
                                                                                         uint64_t q) const {
    static_assert((Ell & (Ell - 1)) == 0);
    static_assert(Ell >= 2);

    if constexpr (Ell == 2) {
        return (prefix_len(offset + 1) < q);
    } else {
        constexpr uint32_t half = Ell / 2;
        const uint32_t step = (prefix_len(offset + half) < q) * half;
        return step + templated_cmov<half>(offset + step, q);
    }
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_array_node<NodePtr, TargetArity>::prefix_before(const std::array<uint64_t, child_capacity>& p,
                                                                 uint32_t i) {
    return i == 0 ? 0 : p[i - 1];
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_array_node<NodePtr, TargetArity>::prefix_len(uint32_t i) const {
    return prefix_before(child_size_, i);
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_array_node<NodePtr, TargetArity>::weight_upperbound(uint64_t level, uint32_t width) {
    uint64_t bound = NodePtr::max_leaf_size(width);
    for (uint64_t i = 0; i < level; ++i) {
        bound *= TargetArity;
    }
    return bound;
}

template <class NodePtr, uint32_t TargetArity>
uint64_t dynamic_array_node<NodePtr, TargetArity>::weight_lowerbound(uint64_t level, uint32_t width) {
    return weight_upperbound(level, width) / 4;
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_array_node<NodePtr, TargetArity>::node_overflow(uint32_t k, uint64_t level, uint32_t width) const {
    return children_[k].size() >= weight_upperbound(level, width);
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_array_node<NodePtr, TargetArity>::node_underflow(uint32_t k, uint64_t level, uint32_t width) const {
    return children_[k].size() < weight_lowerbound(level, width);
}

template <class NodePtr, uint32_t TargetArity>
void dynamic_array_node<NodePtr, TargetArity>::split_child(uint32_t k, uint32_t width) {
    NodePtr l;
    NodePtr r;
    children_[k].split_into(&l, &r, width);

    ADAPTIVE_ASSUME(static_cast<uint32_t>(child_count_) + 1 <= child_capacity);
    std::move_backward(children_.begin() + k + 1, children_.begin() + child_count_,
                       children_.begin() + child_count_ + 1);
    children_[k] = std::move(l);
    children_[k + 1] = std::move(r);
    ++child_count_;
    recompute_metadata();
}

template <class NodePtr, uint32_t TargetArity>
bool dynamic_array_node<NodePtr, TargetArity>::merge_children(uint32_t l, uint32_t r, uint64_t level, uint32_t width) {
    return children_[l].merge_with(&children_[r], level, width);
}

template <class NodePtr, uint32_t TargetArity> void dynamic_array_node<NodePtr, TargetArity>::recompute_metadata() {
    uint64_t size_prefix = 0;
    for (uint32_t k = 0; k < child_count_; ++k) {
        size_prefix += children_[k].size();
        child_size_[k] = size_prefix;
    }
    for (uint32_t k = child_count_; k < child_capacity; ++k) {
        children_[k].reset();
        child_size_[k] = std::numeric_limits<uint64_t>::max();
    }
    size_ = size_prefix;
    last_ = (child_count_ > 0 && size_ > 0) ? children_[child_count_ - 1].last() : 0;
}

} // namespace adaptive
