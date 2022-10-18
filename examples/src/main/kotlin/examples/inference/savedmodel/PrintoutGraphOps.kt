/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel

private const val PATH_TO_MODEL = "examples/src/main/resources/models/savedmodel"

/**
 * Prints the TensorFlow graph as a sequence of TensorFlow operands.
 */
fun printOutGraphOps() {
    val model = SavedModel.load(PATH_TO_MODEL)

    println(model.kGraph.toString())
}

/** */
fun main(): Unit = printOutGraphOps()
