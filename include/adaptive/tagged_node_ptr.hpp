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

#include "adaptive/assert.hpp"

namespace adaptive {

template <class Derived, class Leaf, class Static, class Dynamic, uintptr_t TagMask = 0x3> class tagged_node_ptr {
  protected:
    enum class node_tag : uintptr_t {
        dynamic = 1,
        stat = 2,
        leaf = 3,
    };

  public:
    tagged_node_ptr() = default;

    tagged_node_ptr(const tagged_node_ptr&) = delete;
    tagged_node_ptr& operator=(const tagged_node_ptr&) = delete;

    tagged_node_ptr(tagged_node_ptr&& other) noexcept : raw_(std::exchange(other.raw_, 0)) {}

    tagged_node_ptr& operator=(tagged_node_ptr&& other) noexcept {
        if (this != &other) {
            reset();
            raw_ = std::exchange(other.raw_, 0);
        }
        return *this;
    }

    ~tagged_node_ptr() {
        reset();
    }

    template <class... Args> static Derived leaf(Args&&... args) {
        return Derived(new Leaf(std::forward<Args>(args)...), node_tag::leaf);
    }

    template <class... Args> static Derived stat(Args&&... args) {
        return Derived(new Static(std::forward<Args>(args)...), node_tag::stat);
    }

    template <class... Args> static Derived dynamic(Args&&... args) {
        return Derived(new Dynamic(std::forward<Args>(args)...), node_tag::dynamic);
    }

    bool empty() const {
        return raw_ == 0;
    }

    uint64_t size() const {
        switch (tag()) {
        case node_tag::leaf:
            return as_leaf().size();
        case node_tag::stat:
            return as_static().size();
        case node_tag::dynamic:
            return as_dynamic().size();
        }
        return 0;
    }

    uint64_t space_bytes() const {
        switch (tag()) {
        case node_tag::leaf:
            return as_leaf().space_bytes();
        case node_tag::stat:
            return as_static().space_bytes();
        case node_tag::dynamic:
            return as_dynamic().space_bytes();
        }
        return 0;
    }

    void reset() {
        if (empty()) {
            return;
        }
        switch (tag()) {
        case node_tag::leaf:
            delete &as_leaf();
            break;
        case node_tag::stat:
            delete &as_static();
            break;
        case node_tag::dynamic:
            delete &as_dynamic();
            break;
        }
        raw_ = 0;
    }

    bool collapse_root() {
        if (tag() != node_tag::dynamic || as_dynamic().child_count() > 1) {
            return false;
        }
        *static_cast<Derived*>(this) = as_dynamic().take_child(0);
        return true;
    }

    Dynamic& as_dynamic() const {
        ADAPTIVE_ASSUME(tag() == node_tag::dynamic);
        return *static_cast<Dynamic*>(payload());
    }

    Leaf& as_leaf() const {
        ADAPTIVE_ASSUME(tag() == node_tag::leaf);
        return *static_cast<Leaf*>(payload());
    }

    Static& as_static() const {
        ADAPTIVE_ASSUME(tag() == node_tag::stat);
        return *static_cast<Static*>(payload());
    }

  protected:
    tagged_node_ptr(void* ptr, node_tag tag) {
        const auto raw = reinterpret_cast<uintptr_t>(ptr);
        ADAPTIVE_ASSUME(ptr != nullptr);
        ADAPTIVE_ASSUME((raw & TagMask) == 0);
        raw_ = raw | static_cast<uintptr_t>(tag);
    }

    node_tag tag() const {
        ADAPTIVE_ASSUME(!empty());
        return static_cast<node_tag>(raw_ & TagMask);
    }

  private:
    void* payload() const {
        ADAPTIVE_ASSUME(!empty());
        return reinterpret_cast<void*>(raw_ & ~TagMask);
    }

    uintptr_t raw_ = 0;
};

} // namespace adaptive
