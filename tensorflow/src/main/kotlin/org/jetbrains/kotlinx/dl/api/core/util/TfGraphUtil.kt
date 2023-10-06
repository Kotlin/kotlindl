/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Graph
import org.tensorflow.GraphOperation

// TODO: return to KGraph class
internal fun deserializeGraph(graphDef: ByteArray, prefix: String = ""): Graph {
    return Graph().also { tfGraph ->
        if (prefix.isEmpty()) {
            tfGraph.importGraphDef(graphDef)
        } else {
            tfGraph.importGraphDef(graphDef, prefix)
        }
    }
}

internal fun Graph.copy(): Graph {
    return deserializeGraph(toGraphDef())
}

internal fun Graph.convertToString(): String {
    val operations = operations()

    val sb = StringBuilder()
    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        sb.append("Name: ")
            .append(operation.name())
            .append("; Type: ")
            .append(operation.type())
            .append("; Out #tensors:  ")
            .append(operation.numOutputs())
            .append("\n")
    }
    return sb.toString()
}

internal fun Graph.variableNames(): List<String> {
    val operations = operations()
    val variableNames = mutableListOf<String>()

    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        if (operation.type().equals("VariableV2")) {
            variableNames.add(operation.name())
        }
    }
    return variableNames.toList()
}
