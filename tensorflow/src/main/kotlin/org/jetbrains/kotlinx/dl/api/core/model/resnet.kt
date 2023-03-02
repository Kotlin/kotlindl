/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.model

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding2D
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST

/**
 * Instantiates the ResNet50 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet50-function">
 *    Detailed description of ResNet'50 model and an approach to build it in Keras.</a>
 */
public fun resnet50(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
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
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = false
    )
}

/**
 * Instantiates the ResNet101 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet101-function">
 *    Detailed description of ResNet101 model and an approach to build it in Keras.</a>
 */
public fun resnet101(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack1(x, 64, 3, stride1 = 1, name = "conv2")
        x = stack1(x, 128, 4, name = "conv3")
        x = stack1(x, 256, 23, name = "conv4")
        return stack1(x, 512, 3, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = false
    )
}

/**
 * Instantiates the ResNet152 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet152-function">
 *    Detailed description of ResNet152 model and an approach to build it in Keras.</a>
 */
public fun resnet152(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack1(x, 64, 3, stride1 = 1, name = "conv2")
        x = stack1(x, 128, 8, name = "conv3")
        x = stack1(x, 256, 36, name = "conv4")
        return stack1(x, 512, 3, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = false
    )
}

/**
 * Instantiates the ResNet50V2 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet50v2-function">
 *    Detailed description of ResNet50V2 model and an approach to build it in Keras.</a>
 */
public fun resnet50v2(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack2(x, 64, 3, name = "conv2")
        x = stack2(x, 128, 4, name = "conv3")
        x = stack2(x, 256, 6, name = "conv4")
        return stack2(x, 512, 3, stride1 = 1, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = true
    )
}

/**
 * Instantiates the ResNet101V2 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet101v2-function">
 *    Detailed description of ResNet101V2 model and an approach to build it in Keras.</a>
 */
public fun resnet101v2(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack2(x, 64, 3, name = "conv2")
        x = stack2(x, 128, 4, name = "conv3")
        x = stack2(x, 256, 23, name = "conv4")
        return stack2(x, 512, 3, stride1 = 1, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = true
    )
}

/**
 * Instantiates the ResNet152V2 architecture as a Functional model.
 *
 * @param [imageSize] Height = width of image.
 * @param [numberOfClasses] Number of neurons in the last layer (usually, Dense layer).
 * @param [lastLayerActivation] Activation for last layer (usually, Dense layer).
 *
 * @see <a href="https://arxiv.org/abs/1512.03385">
 *     Deep Residual Learning for Image Recognition.</a>
 * @see <a href="https://keras.io/api/applications/resnet/#resnet152v2-function">
 *    Detailed description of ResNet152V2 model and an approach to build it in Keras.</a>
 */
public fun resnet152v2(
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
): Functional {
    val stackFn = fun(pointer: Layer): Layer {
        var x = pointer
        x = stack2(x, 64, 3, name = "conv2")
        x = stack2(x, 128, 8, name = "conv3")
        x = stack2(x, 256, 36, name = "conv4")
        return stack2(x, 512, 3, stride1 = 1, name = "conv5")
    }
    return resnet(
        stackFn = stackFn,
        imageSize = imageSize,
        numberOfClasses = numberOfClasses,
        numberOfInputChannels = numberOfInputChannels,
        lastLayerActivation = lastLayerActivation,
        preact = true
    )
}

/**
 * A set of stacked residual blocks.
 *
 *  @param [pointer]: input tensor.
 *  @param [filters]: filters of the bottleneck layer in a block.
 *  @param [blocks]: blocks in the stacked blocks.
 *  @param [stride1]: default 2, stride of the first layer in the first block.
 *  @param [name]: string, stack label.
 */
private fun stack1(
    pointer: Layer,
    filters: Int,
    blocks: Int,
    stride1: Int = 2,
    name: String,
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

/**
 * A set of stacked residual blocks.
 *
 *  @param [pointer]: input tensor.
 *  @param [filters]: filters of the bottleneck layer in a block.
 *  @param [blocks]: blocks in the stacked blocks.
 *  @param [stride1]: default 2, stride of the first layer in the first block.
 *  @param [name]: string, stack label.
 */
private fun stack2(
    pointer: Layer,
    filters: Int,
    blocks: Int,
    stride1: Int = 2,
    name: String
): Layer {
    var x = pointer

    x = block2(x, filters, convShortcut = true, stride = stride1, name = name + "_block1")
    for (i in 2 until blocks) {
        x = block2(
            x,
            filters,
            name = name + "_block" + i
        )
    }
    x = block2(x, filters, stride = stride1, name = name + "_block" + blocks)
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
            filters = 4 * filters,
            kernelSize = intArrayOf(1, 1),
            strides = intArrayOf(1, stride, stride, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Linear,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.VALID,
            name = name + "_0_conv"
        )(pointer)

        shortcut = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_0_bn")(shortcut)
    } else {
        shortcut = pointer
    }

    var x: Layer = Conv2D(
        filters = filters,
        kernelSize = intArrayOf(1, 1),
        strides = intArrayOf(1, stride, stride, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = name + "_1_conv"
    )(pointer)
    x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_1_bn")(x)
    x = ActivationLayer(activation = Activations.Relu, name = name + "_1_relu")(x)

    x = Conv2D(
        filters = filters,
        kernelSize = intArrayOf(kernelSize, kernelSize),
        strides = intArrayOf(1, 1, 1, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME,
        name = name + "_2_conv"
    )(x)
    x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_2_bn")(x)
    x = ActivationLayer(activation = Activations.Relu, name = name + "_2_relu")(x)

    x = Conv2D(
        filters = 4 * filters,
        kernelSize = intArrayOf(1, 1),
        strides = intArrayOf(1, 1, 1, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = name + "_3_conv"
    )(x)
    x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = name + "_3_bn")(x)

    x = Add(name = name + "_add")(shortcut, x)
    x = ActivationLayer(activation = Activations.Relu, name = name + "_out")(x)

    return x
}

private fun block2(
    pointer: Layer,
    filters: Int,
    kernelSize: Int = 3,
    convShortcut: Boolean = false,
    name: String,
    stride: Int = 1
): Layer {
    var x = pointer
    val bnAxis = listOf(3)
    var preact = BatchNorm(axis = bnAxis, epsilon = 1e-5, name = name + "_preact_bn")(x)
    preact = ActivationLayer(activation = Activations.Relu, name = name + "_preact_relu")(preact)

    val shortcut: Layer

    if (convShortcut) {
        shortcut = Conv2D(
            filters = 4 * filters,
            kernelSize = intArrayOf(1, 1),
            strides = intArrayOf(1, stride, stride, 1),
            dilations = intArrayOf(1, 1, 1, 1),
            activation = Activations.Linear,
            kernelInitializer = GlorotUniform(),
            biasInitializer = Zeros(),
            padding = ConvPadding.VALID,
            name = name + "_0_conv"
        )(preact)
    } else {
        shortcut = if (stride > 1) {
            return MaxPool2D(
                poolSize = intArrayOf(1, 3, 3, 1),
                strides = intArrayOf(1, 2, 2, 1),
                padding = ConvPadding.VALID,
                name = "pool1_pool"
            )(x)
        } else x

    }

    x = Conv2D(
        filters = filters,
        kernelSize = intArrayOf(1, 1),
        strides = intArrayOf(1, 1, 1, 1),
        useBias = false,
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = name + "_1_conv"
    )(preact)

    x = BatchNorm(axis = bnAxis, epsilon = 1e-5, name = name + "_1_bn")(x)

    x = ActivationLayer(activation = Activations.Relu, name = name + "_1_relu")(x)

    x = ZeroPadding2D(intArrayOf(1, 1, 1, 1), dataFormat = CHANNELS_LAST, name = name + "_2_pad")(x)

    x = Conv2D(
        filters = filters,
        kernelSize = intArrayOf(kernelSize, kernelSize),
        strides = intArrayOf(1, stride, stride, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        useBias = true,
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = name + "_2_conv"
    )(x)
    x = BatchNorm(axis = bnAxis, epsilon = 1e-5, name = name + "_2_bn")(x)
    x = ActivationLayer(activation = Activations.Relu, name = name + "_2_relu")(x)

    x = Conv2D(
        filters = 4 * filters,
        kernelSize = intArrayOf(1, 1),
        strides = intArrayOf(1, 1, 1, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = name + "_3_conv"
    )(x)

    x = Add(name = name + "_out")(shortcut, x)

    return x
}

private fun resnet(
    stackFn: (x: Layer) -> Layer,
    imageSize: Long = 224,
    numberOfClasses: Int = 10,
    numberOfInputChannels: Long = 3,
    lastLayerActivation: Activations = Activations.Linear,
    /** whether to use pre-activation or not
    (True for ResNetV2, False for ResNet and ResNeXt). */
    preact: Boolean
): Functional {
    var x: Layer = Input(
        imageSize,
        imageSize,
        numberOfInputChannels,
        name = "input_1"
    )

    x = ZeroPadding2D(intArrayOf(3, 3, 3, 3), dataFormat = CHANNELS_LAST, name = "conv1_pad")(x)

    x = Conv2D(
        filters = 64,
        kernelSize = intArrayOf(7, 7),
        strides = intArrayOf(1, 2, 2, 1),
        dilations = intArrayOf(1, 1, 1, 1),
        activation = Activations.Linear,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        padding = ConvPadding.VALID,
        name = "conv1_conv"
    )(x)

    if (!preact) {
        x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = "conv1_bn")(x)
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
        x = BatchNorm(axis = listOf(3), epsilon = 1e-5, name = "post_bn")(x)
        x = ActivationLayer(activation = Activations.Relu, name = "post_relu")(x)
    }

    x = GlobalAvgPool2D(name = "avg_pool")(x)
    x = Dense(
        outputSize = numberOfClasses,
        activation = lastLayerActivation,
        kernelInitializer = GlorotUniform(),
        biasInitializer = Zeros(),
        name = "predictions"
    )(x)

    return Functional.of(x)
}


