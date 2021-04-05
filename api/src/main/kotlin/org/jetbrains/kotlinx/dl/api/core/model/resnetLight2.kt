/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.model

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding2D
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST

public fun resnet50Light2(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack1(x, 64, 3, stride1 = 1, name = "conv2")
        x = stack1(x, 128, 4, name = "conv3")
        x = stack1(x, 256, 6, name = "conv4")
        return stack1(x, 512, 3, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        lastLayerActivation = lastLayerActivation,
        preact = false
    )
}


/**
 * A set of stacked residual blocks.
 *
 *  x: input tensor.
filters: integer, filters of the bottleneck layer in a block.
blocks: integer, blocks in the stacked blocks.
stride1: default 2, stride of the first layer in the first block.
name: string, stack label.
 */
private fun stack1(
    pointer: Layer,
    filters: Int,
    blocks: Int,
    stride1: Int = 2,
    name: String
): Layer {
    var x = pointer

    x = block1(x, filters, stride = stride1, name = name + "_block1")
    for (i in 2 until blocks + 1) {
        x = block1(
            x,
            filters,
            conv_shortcut = false,
            name = name + "_block" + i
        )
    }
    return x
}


private fun block1(
    pointer: Layer,
    filters: Int,
    kernelSize: Int = 3,
    conv_shortcut: Boolean = true,
    name: String,
    stride: Int = 1
): Layer {
    var shortcut: Layer

    if (conv_shortcut) {
        shortcut = Conv2D(
            filters = (4 * filters).toLong(),
            kernelSize = longArrayOf(1, 1),
            strides = longArrayOf(1, stride.toLong(), stride.toLong(), 1),
            dilations = longArrayOf(1, 1, 1, 1),
            activation = Activations.Linear,
            kernelInitializer = HeNormal(),
            biasInitializer = HeNormal(),
            padding = ConvPadding.VALID,
            name = name + "_0_conv"
        )(pointer)

        /* shortcut = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_0_bn")(shortcut)
         layerList.add(shortcut)*/
    } else {
        shortcut = pointer
    }

    var x: Layer = Conv2D(
        filters = filters.toLong(),
        kernelSize = longArrayOf(1, 1),
        strides = longArrayOf(1, stride.toLong(), stride.toLong(), 1),
        dilations = longArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        padding = ConvPadding.VALID,
        name = name + "_1_conv"
    )(pointer)

    /* x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_1_bn")(x)
     layerList.add(x)*/
    x = ActivationLayer(activation = Activations.Relu, name = name + "_1_relu")(x)


    x = Conv2D(
        filters = filters.toLong(),
        kernelSize = longArrayOf(kernelSize.toLong(), kernelSize.toLong()),
        strides = longArrayOf(1, 1, 1, 1),
        dilations = longArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        padding = ConvPadding.SAME,
        name = name + "_2_conv"
    )(x)

    /*x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_2_bn")(x)
    layerList.add(x)*/
    x = ActivationLayer(activation = Activations.Relu, name = name + "_2_relu")(x)


    x = Conv2D(
        filters = (4 * filters).toLong(),
        kernelSize = longArrayOf(1, 1),
        strides = longArrayOf(1, 1, 1, 1),
        dilations = longArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        padding = ConvPadding.VALID,
        name = name + "_3_conv"
    )(x)

    /*x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_3_bn")(x)
    layerList.add(x)*/

    x = Add(name = name + "_add")(shortcut, x)

    x = ActivationLayer(activation = Activations.Relu, name = name + "_out")(x)


    return x
}


private fun resnet(
    stackFn: (x: Layer) -> Layer,
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    lastLayerActivation: Activations = Activations.Linear,
    /** whether to use pre-activation or not
    (True for ResNetV2, False for ResNet and ResNeXt). */
    preact: Boolean
): Functional {
    var x: Layer = Input(
        imageSize,
        imageSize,
        3,
        name = "input_1"
    )

    x = ZeroPadding2D(intArrayOf(3, 3, 3, 3), dataFormat = CHANNELS_LAST, name = "conv1_pad")(x)

    x = Conv2D(
        filters = 64,
        kernelSize = longArrayOf(7, 7),
        strides = longArrayOf(1, 2, 2, 1),
        dilations = longArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        padding = ConvPadding.VALID,
        name = "conv1_conv"
    )(x)

    if (!preact) {
        /* x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = "conv1_bn")(x)
         layerList.add(x)*/
        x = ActivationLayer(activation = Activations.Relu, name = "conv1_relu")(x)
    }

    x = ZeroPadding2D(intArrayOf(1, 1, 1, 1), dataFormat = CHANNELS_LAST, name = "pool1_pad")(x)
    x = MaxPool2D(
        poolSize = intArrayOf(1, 3, 3, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.VALID,
        name = "pool1_pool"
    )(x)

    x = stackFn(x)

    if (preact) {
        /*x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = "post_bn")(x)
        layerList.add(x)*/
        x = ActivationLayer(activation = Activations.Relu, name = "post_relu")(x)
    }

    x = GlobalAvgPool2D(name = "avg_pool")(x)
    x = Dense(
        outputSize = numberOfClasses,
        activation = lastLayerActivation,
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        name = "predictions"
    )(x)

    return Functional.fromFinalLayer(x)
}


