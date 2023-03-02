/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.models

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.summary.LayerSummary
import org.jetbrains.kotlinx.dl.api.core.summary.TfModelSummary
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 13L

internal class FunctionalModelTest {
    private val correctTestModel = Functional.of(
        input,
        conv2D_1(input),
        conv2D_2(conv2D_1),
        maxPool2D(conv2D_2),
        conv2D_4(maxPool2D),
        conv2D_5(conv2D_4),
        add(conv2D_5, maxPool2D),
        conv2D_6(add),
        conv2D_7(conv2D_6),
        add_1(conv2D_7, add),
        conv2D_8(add_1),
        globalAvgPool2D(conv2D_8),
        dense_1(globalAvgPool2D),
        dense_2(dense_1)
    ).apply {
        name = "functional_model"
    }

    private val correctTestModelWithUnsortedLayers = Functional.of(
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
    ).apply {
        name = "functional_model"
    }

    @Test
    fun summary() {
        dense_2.freeze()

        checkFunctionalModelAfterCompilation(correctTestModel)
        checkFunctionalModelAfterCompilation(correctTestModelWithUnsortedLayers)
    }

    private fun checkFunctionalModelAfterCompilation(model: Functional) {
        model.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Accuracy()
        )

        assertEquals("functional_model", model.name)

        assertEquals(
            TfModelSummary(
                type = "Functional",
                name = "functional_model",
                layersSummaries = listOf(
                    LayerSummary("input_1", "Input", TensorShape(-1, 28, 28, 1), 0, emptyList()),
                    LayerSummary("conv2D_1", "Conv2D", TensorShape(-1, 26, 26, 32), 320, listOf("input_1")),
                    LayerSummary("conv2D_2", "Conv2D", TensorShape(-1, 24, 24, 64), 18496, listOf("conv2D_1")),
                    LayerSummary("maxPool2D", "MaxPool2D", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_2")),
                    LayerSummary("conv2D_4", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("maxPool2D")),
                    LayerSummary("conv2D_5", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("conv2D_4")),
                    LayerSummary("add", "Add", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_5", "maxPool2D")),
                    LayerSummary("conv2D_6", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("add")),
                    LayerSummary("conv2D_7", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("conv2D_6")),
                    LayerSummary("add_1", "Add", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_7", "add")),
                    LayerSummary("conv2D_8", "Conv2D", TensorShape(-1, 6, 6, 64), 36928, listOf("add_1")),
                    LayerSummary("globalAvgPool2D", "GlobalAvgPool2D", TensorShape(-1, 64), 0, listOf("conv2D_8")),
                    LayerSummary("dense_1", "Dense", TensorShape(-1, 256), 16640, listOf("globalAvgPool2D")),
                    LayerSummary("dense_2", "Dense", TensorShape(-1, 10), 2570, listOf("dense_1"))
                ),
                trainableParamsCount = 220096,
                frozenParamsCount = 2570
            ),
            model.summary()
        )
    }
}

private val input = Input(
    IMAGE_SIZE,
    IMAGE_SIZE,
    NUM_CHANNELS,
    name = "input_1"
)
internal val conv2D_1 = Conv2D(
    filters = 32,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_1"
)
internal val conv2D_2 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_2"
)
internal val maxPool2D = MaxPool2D(
    poolSize = intArrayOf(1, 3, 3, 1),
    strides = intArrayOf(1, 3, 3, 1),
    padding = ConvPadding.VALID,
    name = "maxPool2D"
)
internal val conv2D_4 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_4"
)
internal val conv2D_5 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_5"
)
internal val add = Add(name = "add")
internal val conv2D_6 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_6"
)
internal val conv2D_7 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.SAME,
    name = "conv2D_7"
)
internal val add_1 = Add(name = "add_1")
internal val conv2D_8 = Conv2D(
    filters = 64,
    kernelSize = intArrayOf(3, 3),
    strides = intArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = HeNormal(SEED),
    padding = ConvPadding.VALID,
    name = "conv2D_8"
)
internal val globalAvgPool2D = GlobalAvgPool2D(name = "globalAvgPool2D")
internal val dense_1 = Dense(
    outputSize = 256,
    activation = Activations.Relu,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = Constant(0.1f),
    name = "dense_1"
)
internal val dense_2 = Dense(
    outputSize = NUMBER_OF_CLASSES,
    activation = Activations.Linear,
    kernelInitializer = HeNormal(SEED),
    biasInitializer = Constant(0.1f),
    name = "dense_2"
)
