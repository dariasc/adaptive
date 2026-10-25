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
#include <vector>

#include "adaptive/assert.hpp"
#include "adaptive/bit_util.hpp"
#include "adaptive/policy.hpp"
#include "adaptive/tagged_node_ptr.hpp"

namespace adaptive {

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask = 0x3>
class bv_node_ptr
    : public tagged_node_ptr<bv_node_ptr<Leaf, Static, Dynamic, TagMask>, Leaf, Static, Dynamic, TagMask> {
  private:
    using base_type = tagged_node_ptr<bv_node_ptr<Leaf, Static, Dynamic, TagMask>, Leaf, Static, Dynamic, TagMask>;
    using node_tag = typename base_type::node_tag;

  public:
    using base_type::as_dynamic;
    using base_type::as_leaf;
    using base_type::as_static;
    using base_type::base_type;
    using base_type::collapse_root;
    using base_type::dynamic;
    using base_type::empty;
    using base_type::leaf;
    using base_type::reset;
    using base_type::size;
    using base_type::space_bytes;
    using base_type::stat;

    bv_node_ptr() = default;

    bv_node_ptr(const bv_node_ptr&) = delete;
    bv_node_ptr& operator=(const bv_node_ptr&) = delete;

    bv_node_ptr(bv_node_ptr&& other) noexcept = default;
    bv_node_ptr& operator=(bv_node_ptr&& other) noexcept = default;

    ~bv_node_ptr() = default;

    uint64_t ones() const;

    bool access(uint64_t i, const adaptive_policy& policy);
    void read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit);

    template <bool Bit> uint64_t rank(uint64_t i, const adaptive_policy& policy);
    template <bool Bit> uint64_t select(uint64_t j, const adaptive_policy& policy);
    template <bool Bit> int64_t next(uint64_t i, const adaptive_policy& policy);

    int32_t write(uint64_t i, bool value, uint64_t level, bool is_root);
    void insert(uint64_t i, bool value, uint64_t level, bool is_root);
    int32_t erase(uint64_t i, uint64_t level, bool is_root);

    void expand_root();

    static constexpr uint64_t max_leaf_bits();

    void split_into(bv_node_ptr* l, bv_node_ptr* r);
    bool merge_with(bv_node_ptr* r, uint64_t level);

  private:
    using base_type::tag;

    bool record_access(const adaptive_policy& policy);
    void flatten();
    void split_static(uint64_t level, bool is_root);
};

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
constexpr uint64_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::max_leaf_bits() {
    return Leaf::max_bits;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
uint64_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::ones() const {
    switch (tag()) {
    case node_tag::leaf:
        return as_leaf().ones();
    case node_tag::stat:
        return as_static().ones();
    case node_tag::dynamic:
        return as_dynamic().ones();
    }
    return 0;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
bool bv_node_ptr<Leaf, Static, Dynamic, TagMask>::access(uint64_t i, const adaptive_policy& policy) {
    bv_node_ptr* curr = this;
    while (true) {
        switch (curr->tag()) {
        case node_tag::stat:
            return curr->as_static().access(i);
        case node_tag::leaf:
            return curr->as_leaf().access(static_cast<uint32_t>(i));
        case node_tag::dynamic:
            if (curr->record_access(policy)) {
                continue;
            }
            auto& dyn = curr->as_dynamic();
            const uint32_t k = dyn.find_by_offset(i);
            i -= dyn.prefix_len(k);
            curr = &dyn.child(k);
            break;
        }
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::read(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_bit) {
    switch (tag()) {
    case node_tag::stat:
        as_static().read(i, len, out, out_bit);
        return;
    case node_tag::leaf:
        as_leaf().read(static_cast<uint32_t>(i), static_cast<uint32_t>(len), out, out_bit);
        return;
    case node_tag::dynamic:
        as_dynamic().read(i, len, out, out_bit);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
template <bool Bit>
uint64_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::rank(uint64_t i, const adaptive_policy& policy) {
    if constexpr (!Bit) {
        return i - rank<true>(i, policy);
    }
    uint64_t total_rank = 0;
    bv_node_ptr* curr = this;
    while (true) {
        switch (curr->tag()) {
        case node_tag::stat:
            return total_rank + curr->as_static().template rank<true>(i);
        case node_tag::leaf:
            return total_rank + curr->as_leaf().template rank<true>(static_cast<uint32_t>(i));
        case node_tag::dynamic:
            if (curr->record_access(policy)) {
                continue;
            }
            auto& dyn = curr->as_dynamic();
            ADAPTIVE_ASSUME(i <= dyn.size());
            if (i == 0) {
                return total_rank;
            }
            if (i == dyn.size()) {
                return total_rank + dyn.ones();
            }
            const uint32_t k = dyn.find_by_offset(i - 1);
            total_rank += dyn.template prefix_cnt<1>(k);
            i -= dyn.prefix_len(k);
            curr = &dyn.child(k);
            break;
        }
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
template <bool Bit>
uint64_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::select(uint64_t j, const adaptive_policy& policy) {
    uint64_t total_pos = 0;
    bv_node_ptr* curr = this;
    while (true) {
        switch (curr->tag()) {
        case node_tag::stat:
            return total_pos + curr->as_static().template select<Bit>(j);
        case node_tag::leaf:
            return total_pos + curr->as_leaf().template select<Bit>(static_cast<uint32_t>(j));
        case node_tag::dynamic:
            if (curr->record_access(policy)) {
                continue;
            }
            auto& dyn = curr->as_dynamic();
            const uint32_t k = dyn.template find_by_cnt<Bit>(j);
            total_pos += dyn.prefix_len(k);
            j -= dyn.template prefix_cnt<Bit>(k);
            curr = &dyn.child(k);
            break;
        }
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
template <bool Bit>
int64_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::next(uint64_t i, const adaptive_policy& policy) {
    switch (tag()) {
    case node_tag::stat:
        return as_static().template next<Bit>(i);
    case node_tag::leaf:
        return as_leaf().template next<Bit>(static_cast<uint32_t>(i));
    case node_tag::dynamic:
        if (record_access(policy)) {
            return next<Bit>(i, policy);
        }
        return as_dynamic().template next<Bit>(i, policy);
    }
    return -1;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
int32_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::write(uint64_t i, bool value, uint64_t level, bool is_root) {
    switch (tag()) {
    case node_tag::leaf:
        return as_leaf().write(static_cast<uint32_t>(i), value);
    case node_tag::stat:
        split_static(level, is_root);
        return write(i, value, level, is_root);
    case node_tag::dynamic:
        return as_dynamic().write(i, value, level);
    }
    return 0;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::insert(uint64_t i, bool value, uint64_t level, bool is_root) {
    switch (tag()) {
    case node_tag::leaf:
        as_leaf().insert(static_cast<uint32_t>(i), value);
        return;
    case node_tag::stat:
        split_static(level, is_root);
        insert(i, value, level, is_root);
        return;
    case node_tag::dynamic:
        as_dynamic().insert(i, value, level);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
int32_t bv_node_ptr<Leaf, Static, Dynamic, TagMask>::erase(uint64_t i, uint64_t level, bool is_root) {
    switch (tag()) {
    case node_tag::leaf:
        return as_leaf().erase(static_cast<uint32_t>(i));
    case node_tag::stat:
        split_static(level, is_root);
        return erase(i, level, is_root);
    case node_tag::dynamic:
        return as_dynamic().erase(i, level);
    }
    return 0;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::expand_root() {
    bv_node_ptr l;
    bv_node_ptr r;
    split_into(&l, &r);
    *this = bv_node_ptr::dynamic();
    as_dynamic().add_child(std::move(l));
    as_dynamic().add_child(std::move(r));
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
bool bv_node_ptr<Leaf, Static, Dynamic, TagMask>::record_access(const adaptive_policy& policy) {
    as_dynamic().record_access();
    if (!as_dynamic().must_flatten(policy)) {
        return false;
    }
    flatten();
    return true;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::flatten() {
    if (tag() != node_tag::dynamic) {
        return;
    }
    const uint64_t len = size();
    const uint64_t words = (len + bit_util::word_bits - 1) / bit_util::word_bits;
    std::vector<uint64_t> data(words + 1);
    read(0, len, data.data(), 0);
    reset();
    if (len < Leaf::max_bits) {
        *this = bv_node_ptr::leaf(data.data(), static_cast<uint32_t>(len));
    } else {
        *this = bv_node_ptr::stat(std::move(data), len);
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::split_static(uint64_t level, bool is_root) {
    ADAPTIVE_ASSUME(tag() == node_tag::stat);
    const uint64_t len = as_static().size();
    const uint64_t n = as_static().ones();
    auto dyn = bv_node_ptr::dynamic(as_static().data(), len, n, level, is_root);
    reset();
    *this = std::move(dyn);
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void bv_node_ptr<Leaf, Static, Dynamic, TagMask>::split_into(bv_node_ptr* l, bv_node_ptr* r) {
    ADAPTIVE_ASSUME(tag() != node_tag::stat);
    if (tag() == node_tag::leaf) {
        ADAPTIVE_ASSUME(size() == Leaf::max_bits);
        const uint64_t bits = Leaf::max_bits / 2;
        const uint64_t* data = as_leaf().data();
        *l = bv_node_ptr::leaf(data, 0, static_cast<uint32_t>(bits));
        *r = bv_node_ptr::leaf(data, bits, static_cast<uint32_t>(bits));
        reset();
        return;
    }
    as_dynamic().split_into(l, r);
    reset();
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
bool bv_node_ptr<Leaf, Static, Dynamic, TagMask>::merge_with(bv_node_ptr* r, uint64_t level) {
    if (tag() == node_tag::stat) {
        split_static(level, false);
    }
    if (r->tag() == node_tag::stat) {
        r->split_static(level, false);
    }
    ADAPTIVE_ASSUME(tag() == r->tag());

    if (tag() == node_tag::leaf) {
        ADAPTIVE_ASSUME(level == 0);
        const uint64_t left_len = size();
        const uint64_t len = left_len + r->size();
        const uint64_t words = (len + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> data(words + 1);
        read(0, left_len, data.data(), 0);
        r->read(0, r->size(), data.data(), left_len);
        if (len < Dynamic::weight_upperbound(level)) {
            r->reset();
            reset();
            *this = bv_node_ptr::leaf(data.data(), static_cast<uint32_t>(len));
            return true;
        }

        const uint64_t l_len = len / 2;
        const uint64_t r_len = len - l_len;
        auto l_node = bv_node_ptr::leaf(data.data(), 0, static_cast<uint32_t>(l_len));
        auto r_node = bv_node_ptr::leaf(data.data(), l_len, static_cast<uint32_t>(r_len));
        *this = std::move(l_node);
        *r = std::move(r_node);
        return false;
    }

    as_dynamic().append_children_from(&r->as_dynamic());
    r->reset();
    return true;
}

} // namespace adaptive
