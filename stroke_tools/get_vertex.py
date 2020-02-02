# -*- coding: utf-8 -*-
"""
Copyright (C) 2006 Kouhei Sutou <kou@cozmixng.org>
(converted to python from C by David Morrissey)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 59 Temple Place, Suite 330,
Boston, MA  02111-1307  USA

$Id: tomoe-recognizer-simple-logic.c 917 2006-11-30 09:58:39Z ikezoe $
"""


def get_distance(nodes, first_node_idx, last_node_idx):
    """
    Getting distance
    MAX( |aw - bv + c| )
    a = x-p : b = y-q : c = py - qx
    first = (p, q) : last = (x, y) : other = (v, w)
    """
    max = 0
    first = nodes[first_node_idx]
    last = nodes[last_node_idx]

    most_node = None
    if first_node_idx == last_node_idx:
        return 0, most_node

    a = last[0] - first[0]
    b = last[1] - first[1]
    c = last[1] * first[0] - last[0] * first[1]

    for node_idx in range(first_node_idx, last_node_idx+1):  # OFF BY ONE ERROR?
        node = nodes[node_idx]
        dist = abs((a * node[1]) - (b * node[0]) + c)
        if dist > max:
            max = dist
            most_node = node_idx

    denom = a * a + b * b
    if denom == 0:
        return 0, most_node
    else:
        return max * max // denom, most_node


TOMOE_WRITING_WIDTH = 1000


def _get_vertex(nodes, first_node_idx, last_node_idx):
    rv = []
    ERROR = TOMOE_WRITING_WIDTH * TOMOE_WRITING_WIDTH // 4444  # 5%

    dist, most_node_idx = get_distance(nodes, first_node_idx, last_node_idx)
    if dist > ERROR:
        rv = _get_vertex(nodes, first_node_idx, most_node_idx) + \
             _get_vertex(nodes, most_node_idx, last_node_idx)
    else:
        rv.append(nodes[last_node_idx])
    return rv


def get_vertex(nodes):
    L = [nodes[0]]
    LExtend = _get_vertex(nodes, 0, len(nodes)-1)
    if LExtend[0] != L[0]:
        L.extend(LExtend)
    return L


if __name__ == '__main__':
    L = [[54, 19], [53, 19], [44, 29], [30, 47], [23, 58], [21, 60], [19, 63]]
    L = [[(0, 0), (5, 12), (11, 141), (17, 282), (17, 442), (17, 564), (17, 673), (17, 743), (17, 801), (17, 839), (23, 871), (28, 891), (28, 884), (40, 884), (138, 903), (323, 935), (566, 967), (820, 993), (965, 1000), (1000, 1000)]]

    for i in L:
        print(i)
        print(get_vertex(i))

    [[(208, 0), (199, 119), (94, 341)], [(0, 461), (781, 520), (915, 520), (999, 479)],
                [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)], [(303, 514), (94, 766)],
                [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)], [(716, 628), (462, 916)],
                [(696, 101), (771, 155), (835, 251)]]
