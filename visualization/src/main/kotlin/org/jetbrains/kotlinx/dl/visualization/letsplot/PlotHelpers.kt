/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.intern.Plot

internal typealias TensorImageData = Array<Array<Array<FloatArray>>>

internal fun inputsOutputsPlots(inputs: Int, outputs: Int, f: (Int, Int) -> Plot) =
    List(inputs) { i -> List(outputs) { o -> f(i, o) } }.flatten()

internal fun extractXYInputOutputAxeSizes(inputData: TensorImageData, permute: IntArray = intArrayOf(0, 1, 2, 3)): IntArray =
    with(IntArray(4)) {
        this[permute[0]] = inputData.size // default xSize
        this[permute[1]] = inputData[0].size // default ySize
        this[permute[2]] = inputData[0][0].size // default inputs
        this[permute[3]] = inputData[0][0][0].size // default outputs
        this
    }
