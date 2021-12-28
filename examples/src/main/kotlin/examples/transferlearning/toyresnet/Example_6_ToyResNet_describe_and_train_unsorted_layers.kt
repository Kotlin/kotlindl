/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import org.jetbrains.kotlinx.dl.dataset.handler.NUMBER_OF_CLASSES

/**
 * What's about Functional API usage in KotlinDL directly?
 *
 * Describe the model like the function of functions, where each layer is just a callable function.
 *
 * Combine two functions in special merge layers like Add or Concatenate.
 *
 * NOTE: Functional API supports one output and one input for model.
 */
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 13L

private val input = Input(
    IMAGE_SIZE,
    IMAGE_SIZE,
    NUM_CHANNELS,
    name = "input_1"
)
private val conv2D_1 = Conv2D(
    filters = 32,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_1"
)
private val conv2D_2 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_2"
)
private val maxPool2D = MaxPool2D(
    poolSize = intArrayOf(1, 3, 3, 1),
    strides = intArrayOf(1, 3, 3, 1),
    padding = ConvPadding.VALID,
    name = "maxPool2D"
)
private val conv2D_4 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_4"
)
private val conv2D_5 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_5"
)
private val add = Add(name = "add")
private val conv2D_6 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_6"
)
private val conv2D_7 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_7"
)
private val add_1 = Add(name = "add_1")
private val conv2D_8 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_8"
)
private val globalAvgPool2D = GlobalAvgPool2D(name = "globalAvgPool2D")
private val dense_1 = Dense(
    outputSize = 256,
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = Constant(0.1f),
    name = "dense_1"
)
private val dense_2 = Dense(
    outputSize = NUMBER_OF_CLASSES,
    activation = Activations.Linear,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = Constant(0.1f),
    name = "dense_2"
)

// TODO: move to tests
private val model = Functional.of(
    input,
    conv2D_2(conv2D_1),
    maxPool2D(conv2D_2),
    globalAvgPool2D(conv2D_8),
    dense_1(globalAvgPool2D),
    conv2D_1(input),
    conv2D_4(maxPool2D),
    conv2D_5(conv2D_4),
    add(conv2D_5, maxPool2D),
    conv2D_6(add),
    conv2D_7(conv2D_6),
    add_1(conv2D_7, add),
    conv2D_8(add_1),
    dense_2(dense_1)
)

fun main() {
    val (train, test) = fashionMnist()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.init()
        var accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after: $accuracy")
    }
}

