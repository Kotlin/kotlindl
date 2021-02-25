/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.production

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.NUMBER_OF_CLASSES

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val kernelInitializer = GlorotNormal(SEED)
private val biasInitializer = GlorotUniform(SEED)

val lenet5 = Sequential.of(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS,
        name = "input_0"
    ),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_1"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_2"
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_3"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_4"
    ),
    Flatten(name = "flatten_5"), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_6"
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_7"
    ),
    Dense(
        outputSize = NUMBER_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_8"
    )
)

fun getLabel(dataset: Dataset, imageId: Int): Int {
    val imageLabel = dataset.getY(imageId)
    return imageLabel.indexOfFirst { it == imageLabel.maxOrNull()!! }
}

fun mnistReshape(image: FloatArray): Array<Array<Array<FloatArray>>> {
    val reshaped = Array(
        1
    ) { Array(28) { Array(28) { FloatArray(1) } } }
    for (i in image.indices) reshaped[0][i / 28][i % 28][0] = image[i]
    return reshaped
}
