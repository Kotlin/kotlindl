/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.models

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten

/**
 * This is an CNN based on an implementation of LeNet-5 from classic paper.
 *
 * @see <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">
 *    Gradient-based learning applied to document recognition:[Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, 1998]</a>
 */
internal fun buildLetNet5Classic(
    imageWidth: Long,
    imageHeight: Long,
    numChannels: Long,
    numClasses: Int,
    layersActivation: Activations,
    classifierActivation: Activations,
    randomSeed: Long,
): Sequential {
    return Sequential.of(
        Input(
            imageWidth,
            imageHeight,
            numChannels,
        ),
        Conv2D(
            filters = 6,
            kernelSize = 5,
            strides = 1,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(randomSeed),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
        ),
        AvgPool2D(
            poolSize = 2,
            strides = 2,
            padding = ConvPadding.VALID,
        ),
        Conv2D(
            filters = 16,
            kernelSize = 5,
            strides = 1,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(randomSeed),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
        ),
        AvgPool2D(
            poolSize = 2,
            strides = 2,
            padding = ConvPadding.VALID,
        ),
        Flatten(),
        Dense(
            outputSize = 120,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(randomSeed),
            biasInitializer = Constant(0.1f),
        ),
        Dense(
            outputSize = 84,
            activation = Activations.Tanh,
            kernelInitializer = GlorotNormal(randomSeed),
            biasInitializer = Constant(0.1f),
        ),
        Dense(
            outputSize = numClasses,
            activation = classifierActivation,
            kernelInitializer = GlorotNormal(randomSeed),
            biasInitializer = Constant(0.1f),
        )
    )
}
