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
class array_node_ptr
    : public tagged_node_ptr<array_node_ptr<Leaf, Static, Dynamic, TagMask>, Leaf, Static, Dynamic, TagMask> {
  private:
    using base_type = tagged_node_ptr<array_node_ptr<Leaf, Static, Dynamic, TagMask>, Leaf, Static, Dynamic, TagMask>;
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

    array_node_ptr() = default;

    array_node_ptr(const array_node_ptr&) = delete;
    array_node_ptr& operator=(const array_node_ptr&) = delete;

    array_node_ptr(array_node_ptr&& other) noexcept = default;
    array_node_ptr& operator=(array_node_ptr&& other) noexcept = default;

    ~array_node_ptr() = default;

    uint64_t access(uint64_t i, uint32_t width, const adaptive_policy& policy);
    uint64_t last() const;
    std::pair<uint64_t, uint64_t> next(uint64_t i, uint64_t j, uint64_t c, uint32_t width,
                                       const adaptive_policy& policy);
    void read_packed(uint64_t i, uint64_t len, uint64_t* out, uint64_t out_index, uint32_t width);

    void write(uint64_t i, uint64_t value, uint64_t level, bool is_root, uint32_t width);
    void insert(uint64_t i, uint64_t value, uint64_t level, bool is_root, uint32_t width);
    void erase(uint64_t i, uint64_t level, bool is_root, uint32_t width);

    void expand_root(uint32_t width);

    static uint32_t max_leaf_size(uint32_t width);

    void split_into(array_node_ptr* l, array_node_ptr* r, uint32_t width);
    bool merge_with(array_node_ptr* r, uint64_t level, uint32_t width);

  private:
    using base_type::tag;

    bool record_access(uint32_t width, const adaptive_policy& policy);
    void flatten(uint32_t width);
    void split_static(uint64_t level, bool is_root, uint32_t width);
};

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
uint32_t array_node_ptr<Leaf, Static, Dynamic, TagMask>::max_leaf_size(uint32_t width) {
    return Leaf::max_size(width);
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
uint64_t array_node_ptr<Leaf, Static, Dynamic, TagMask>::last() const {
    switch (tag()) {
    case node_tag::leaf:
        return as_leaf().last();
    case node_tag::stat:
        return as_static().last();
    case node_tag::dynamic:
        return as_dynamic().last();
    }
    return 0;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
uint64_t array_node_ptr<Leaf, Static, Dynamic, TagMask>::access(uint64_t i, uint32_t width,
                                                                const adaptive_policy& policy) {
    array_node_ptr* curr = this;
    while (true) {
        switch (curr->tag()) {
        case node_tag::stat:
            return curr->as_static().access(i);
        case node_tag::leaf:
            return curr->as_leaf().access(static_cast<uint32_t>(i));
        case node_tag::dynamic:
            if (curr->record_access(width, policy)) {
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
std::pair<uint64_t, uint64_t> array_node_ptr<Leaf, Static, Dynamic, TagMask>::next(uint64_t i, uint64_t j, uint64_t c,
                                                                                   uint32_t width,
                                                                                   const adaptive_policy& policy) {
    array_node_ptr* curr = this;
    uint64_t base_offset = 0;
    while (true) {
        switch (curr->tag()) {
        case node_tag::stat: {
            auto res = curr->as_static().next(i, j, c);
            return {res.first, res.second + base_offset};
        }
        case node_tag::leaf: {
            auto res = curr->as_leaf().next(static_cast<uint32_t>(i), static_cast<uint32_t>(j), c);
            return {res.first, static_cast<uint64_t>(res.second) + base_offset};
        }
        case node_tag::dynamic: {
            if (curr->record_access(width, policy)) {
                continue;
            }
            auto& dyn = curr->as_dynamic();
            if (j == dyn.size() - 1 && dyn.last() < c) {
                return {0, j + 1 + base_offset};
            }
            const uint32_t start_k = dyn.find_by_offset(i);
            const uint32_t end_k = dyn.find_by_offset(j);
            if (start_k == end_k) {
                const uint64_t l_len = dyn.prefix_len(start_k);
                i -= l_len;
                j -= l_len;
                base_offset += l_len;
                curr = &dyn.child(start_k);
                break;
            }
            auto res = dyn.next(i, j, c, width, policy);
            return {res.first, res.second + base_offset};
        }
        }
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::read_packed(uint64_t i, uint64_t len, uint64_t* out,
                                                                 uint64_t out_index, uint32_t width) {
    switch (tag()) {
    case node_tag::leaf:
        as_leaf().read_packed(static_cast<uint32_t>(i), static_cast<uint32_t>(len), out, out_index);
        return;
    case node_tag::stat:
        as_static().read_packed(i, len, out, out_index);
        return;
    case node_tag::dynamic:
        as_dynamic().read_packed(i, len, out, out_index, width);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::write(uint64_t i, uint64_t value, uint64_t level, bool is_root,
                                                           uint32_t width) {
    switch (tag()) {
    case node_tag::leaf:
        as_leaf().write(static_cast<uint32_t>(i), value);
        return;
    case node_tag::stat:
        split_static(level, is_root, width);
        write(i, value, level, is_root, width);
        return;
    case node_tag::dynamic:
        as_dynamic().write(i, value, level, width);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::insert(uint64_t i, uint64_t value, uint64_t level, bool is_root,
                                                            uint32_t width) {
    switch (tag()) {
    case node_tag::leaf:
        as_leaf().insert(static_cast<uint32_t>(i), value);
        return;
    case node_tag::stat:
        split_static(level, is_root, width);
        insert(i, value, level, is_root, width);
        return;
    case node_tag::dynamic:
        as_dynamic().insert(i, value, level, width);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::erase(uint64_t i, uint64_t level, bool is_root, uint32_t width) {
    switch (tag()) {
    case node_tag::leaf:
        as_leaf().erase(static_cast<uint32_t>(i));
        return;
    case node_tag::stat:
        split_static(level, is_root, width);
        erase(i, level, is_root, width);
        return;
    case node_tag::dynamic:
        as_dynamic().erase(i, level, width);
        return;
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::expand_root(uint32_t width) {
    array_node_ptr l;
    array_node_ptr r;
    split_into(&l, &r, width);
    *this = array_node_ptr::dynamic();
    as_dynamic().add_child(std::move(l));
    as_dynamic().add_child(std::move(r));
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
bool array_node_ptr<Leaf, Static, Dynamic, TagMask>::record_access(uint32_t width, const adaptive_policy& policy) {
    as_dynamic().record_access();
    if (!as_dynamic().must_flatten(policy)) {
        return false;
    }
    flatten(width);
    return true;
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::flatten(uint32_t width) {
    if (tag() != node_tag::dynamic) {
        return;
    }
    const uint64_t len = size();
    const uint64_t words = (len * width + bit_util::word_bits - 1) / bit_util::word_bits;
    std::vector<uint64_t> data(words + 1);
    read_packed(0, len, data.data(), 0, width);
    reset();
    if (len < Leaf::max_size(width)) {
        *this = array_node_ptr::leaf(data.data(), 0, static_cast<uint32_t>(len), width);
    } else {
        *this = array_node_ptr::stat(std::move(data), len, width);
    }
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::split_static(uint64_t level, bool is_root, uint32_t width) {
    ADAPTIVE_ASSUME(tag() == node_tag::stat);
    const uint64_t len = as_static().size();
    auto dyn = array_node_ptr::dynamic(as_static().data(), len, level, is_root, width);
    reset();
    *this = std::move(dyn);
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
void array_node_ptr<Leaf, Static, Dynamic, TagMask>::split_into(array_node_ptr* l, array_node_ptr* r, uint32_t width) {
    ADAPTIVE_ASSUME(tag() != node_tag::stat);
    if (tag() == node_tag::leaf) {
        const uint64_t left_len = size() / 2;
        const uint64_t right_len = size() - left_len;
        const uint64_t* data = as_leaf().data();
        *l = array_node_ptr::leaf(data, 0, static_cast<uint32_t>(left_len), width);
        *r = array_node_ptr::leaf(data, left_len, static_cast<uint32_t>(right_len), width);
        reset();
        return;
    }
    as_dynamic().split_into(l, r);
    reset();
}

template <class Leaf, class Static, class Dynamic, uintptr_t TagMask>
bool array_node_ptr<Leaf, Static, Dynamic, TagMask>::merge_with(array_node_ptr* r, uint64_t level, uint32_t width) {
    if (tag() == node_tag::stat) {
        split_static(level, false, width);
    }
    if (r->tag() == node_tag::stat) {
        r->split_static(level, false, width);
    }
    ADAPTIVE_ASSUME(tag() == r->tag());

    if (tag() == node_tag::leaf) {
        ADAPTIVE_ASSUME(level == 0);
        const uint64_t left_len = size();
        const uint64_t len = left_len + r->size();
        const uint64_t words = (len * width + bit_util::word_bits - 1) / bit_util::word_bits;
        std::vector<uint64_t> data(words + 1);
        read_packed(0, left_len, data.data(), 0, width);
        r->read_packed(0, r->size(), data.data(), left_len, width);
        if (len < Dynamic::weight_upperbound(level, width)) {
            r->reset();
            reset();
            *this = array_node_ptr::leaf(data.data(), 0, static_cast<uint32_t>(len), width);
            return true;
        }

        const uint64_t l_len = len / 2;
        const uint64_t r_len = len - l_len;
        auto l_node = array_node_ptr::leaf(data.data(), 0, static_cast<uint32_t>(l_len), width);
        auto r_node = array_node_ptr::leaf(data.data(), l_len, static_cast<uint32_t>(r_len), width);
        reset();
        r->reset();
        *this = std::move(l_node);
        *r = std::move(r_node);
        return false;
    }

    as_dynamic().append_children_from(&r->as_dynamic());
    r->reset();
    return true;
}

} // namespace adaptive
