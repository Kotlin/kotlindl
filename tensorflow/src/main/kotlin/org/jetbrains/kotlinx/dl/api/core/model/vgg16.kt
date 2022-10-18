/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.model

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten

/**
 * Instantiates the VGG16 model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
 *    Detailed description of VGG'16 model and an approach to build it in Keras.</a>
 */
public fun vgg16(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear
): Sequential {
    return Sequential.of(
        Input(
            imageSize,
            imageSize,
            numberOfInputChannels
        ),
        // Block #1
        Conv2D(
            filters = 64,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block1_conv1"
        ),
        Conv2D(
            filters = 64,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block1_conv2"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            padding = ConvPadding.VALID,
            name = "block1_pool"
        ),
        // Block #2
        Conv2D(
            filters = 128,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block2_conv1"
        ),
        Conv2D(
            filters = 128,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block2_conv2"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            padding = ConvPadding.VALID,
            name = "block2_pool"
        ),
        // Block #3
        Conv2D(
            filters = 256,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block3_conv1"
        ),
        Conv2D(
            filters = 256,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block3_conv2"
        ),
        Conv2D(
            filters = 256,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block3_conv3"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            padding = ConvPadding.VALID,
            name = "block3_pool"
        ),
        // Block #4
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block4_conv1"
        ),
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block4_conv2"
        ),
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block4_conv3"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            padding = ConvPadding.VALID,
            name = "block4_pool"
        ),
        // Block #5
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block5_conv1"
        ),
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block5_conv2"
        ),
        Conv2D(
            filters = 512,
            kernelSize = intArrayOf(3, 3),
            strides = intArrayOf(1, 1, 1, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "block5_conv3"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            padding = ConvPadding.VALID,
            name = "block5_pool"
        ),
        Flatten(),
        Dense(
            outputSize = 4096,
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            name = "fc1"
        ),
        Dense(
            outputSize = 4096,
            activation = Activations.Relu,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            name = "fc2"
        ),
        Dense(
            outputSize = numberOfClasses,
            activation = lastLayerActivation,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            name = "predictions"
        )
    )
}
