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

#include <cassert>

#ifndef NDEBUG
#define ADAPTIVE_ASSUME(cond) assert(cond)
#else
#define ADAPTIVE_ASSUME(cond) ((cond) ? void() : __builtin_unreachable())
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ADAPTIVE_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ADAPTIVE_ALWAYS_INLINE inline
#endif

#define ADAPTIVE_UNREACHABLE() __builtin_unreachable()
