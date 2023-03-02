/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

/**
 * Topologically sort nodes in the DAG defined by a provided start node and a function returning next nodes.
 * @param [start] a node from which to start the sort
 * @param [nextNodes] a function which returns a collection of next nodes for a given node
 * @return a list of topologically sorted nodes in the graph
 */
internal fun <T> sortTopologically(start: T, nextNodes: (T) -> Collection<T>): List<T> {
    val visited = mutableSetOf<T>()
    val grayStack: Stack<T> = mutableListOf()

    recursiveTopologicalSort(start, grayStack, visited, nextNodes)

    val sortedList = mutableListOf<T>()
    while (grayStack.isNotEmpty())
        sortedList.add(grayStack.pop()!!)

    return sortedList
}

private fun <T> recursiveTopologicalSort(
    currentNode: T,
    stack: Stack<T>,
    visited: MutableSet<T>,
    nextNodes: (T) -> Collection<T>
) {
    visited.add(currentNode)

    val nextNodesList = nextNodes(currentNode)
    for (nextNode in nextNodesList.reversed()) {
        if (!visited.contains(nextNode)) {
            recursiveTopologicalSort(nextNode, stack, visited, nextNodes)
        }
    }
    stack.push(currentNode)
}