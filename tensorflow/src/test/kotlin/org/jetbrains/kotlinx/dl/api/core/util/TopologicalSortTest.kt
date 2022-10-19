/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class TopologicalSortTest {
    @Test
    fun `line dag`() {
        `test topological sort`(1, 4) { start ->
            when (start) {
                1 -> listOf(2)
                2 -> listOf(3)
                3 -> listOf(4)
                else -> emptyList()
            }
        }
    }

    @Test
    fun diamond() {
        `test topological sort`(1, 4) { start ->
            when (start) {
                1 -> listOf(2, 3)
                2 -> listOf(4)
                3 -> listOf(4)
                else -> emptyList()
            }
        }
    }

    @Test
    fun `triple merge`() {
        `test topological sort`(0, 6) { start ->
            when (start) {
                0 -> listOf(1, 2, 4)
                1 -> listOf(4)
                2 -> listOf(3)
                3 -> listOf(5)
                4 -> listOf(5)
                else -> emptyList()
            }
        }
    }

    @Test
    fun cycle() {
        `test topological sort`(1, 3) { start ->
            when (start) {
                1 -> listOf(2)
                2 -> listOf(3)
                3 -> listOf(1)
                else -> emptyList()
            }
        }
    }

    @Test
    fun `disconnected graph`() {
        `test topological sort`(1, 3) { start ->
            when (start) {
                1 -> listOf(2)
                2 -> listOf(4)
                3 -> listOf(4)
                else -> emptyList()
            }
        }
    }

    private fun `test topological sort`(start: Int, size: Int, nextNodes: (Int) -> Collection<Int>) {
        val actuallySorted = sortTopologically(start, nextNodes)
        Assertions.assertEquals(size, actuallySorted.size)
        Assertions.assertEquals(actuallySorted.sorted(), actuallySorted)
    }
}