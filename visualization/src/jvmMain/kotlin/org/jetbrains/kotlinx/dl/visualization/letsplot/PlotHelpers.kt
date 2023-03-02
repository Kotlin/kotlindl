/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

internal typealias TensorImageData = Array<Array<Array<FloatArray>>>

/**
 * Create a list of pairs that represents the cartesian product of numbers
 * from the ranges [0, x) and [0, y).
 *
 * @param x specifying the first range [0, x)
 * @param y specifying the second range [0, y)
 * @return list of pairs of numbers from cartesian product
 */
internal fun cartesianProductIndices(x: Int, y: Int): List<Pair<Int, Int>> =
    List(x) { i -> List(y) { o -> Pair(i, o) } }.flatten()

/**
 * Extract x, y, input, and output axe sizes from the tensor data that
 * is actual data of some weights from the model.
 *
 * @param inputData 4D tensor data representing the weights of some model
 * @param permute array of permutation of the result sizes. Defaults to identity
 * permutation that causes results in (x, y, input, output) sizes in returned sizes
 * @return array with 4 numbers representing the sizes (x, y, input, output) of
 * [inputData] according to given permutation
 */
internal fun extractXYInputOutputAxeSizes(
    inputData: TensorImageData,
    permute: IntArray = intArrayOf(0, 1, 2, 3)
): IntArray =
    with(IntArray(4)) {
        this[permute[0]] = inputData.size // default xSize
        this[permute[1]] = inputData[0].size // default ySize
        this[permute[2]] = inputData[0][0].size // default inputs
        this[permute[3]] = inputData[0][0][0].size // default outputs
        this
    }
