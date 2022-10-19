/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.models

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.exception.RepeatableLayerNameException
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.isTrainable
import org.jetbrains.kotlinx.dl.api.core.layer.paramCount
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val AMOUNT_OF_CLASSES = 10


internal class SequentialModelTest {
    private val correctTestModelLayers = listOf(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        ),
        Conv2D(
            filters = 32,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "conv2d_1"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_1"
        ),
        Conv2D(
            filters = 64,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "conv2d_2"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_2"
        ),
        Flatten(name = "flatten_1"), // 3136
        Dense(
            outputSize = 512,
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_1"
        ),
        Dense(
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_2"
        )
    )

    @Test
    fun buildModel() {
        val correctTestModel = Sequential.of(correctTestModelLayers).apply {
            name = "sequential_model"
        }

        assertEquals(correctTestModel.layers.size, 8)
        assertTrue(correctTestModel.getLayer("conv2d_1") is Conv2D)
        assertTrue(correctTestModel.getLayer("conv2d_2") is Conv2D)
        assertTrue(correctTestModel.getLayer("conv2d_1").isTrainable)
        assertTrue(correctTestModel.getLayer("conv2d_1").hasActivation)
        assertFalse(correctTestModel.getLayer("flatten_1").isTrainable)
        assertFalse(correctTestModel.getLayer("flatten_1").hasActivation)
        assertArrayEquals(correctTestModel.inputLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    /**
     * Kotlin: [Internal Error] org.jetbrains.kotlin.backend.common.BackendException: Backend Internal error: Exception during psi2ir
    File being compiled: (112,21) in C:/Users/zaleslaw/IdeaProjects/KotlinDL/api/src/test/kotlin/org/jetbrains/kotlinx/dl/api/core/models/SequentialCompilationTest.kt
    The root cause java.lang.StackOverflowError was thrown at: java.lang.ClassLoader.defineClass1(Native Method)
    null: KtBinaryExpression:
    "Name: default_data_placeholder; Type: Placeholder; Out #tensors:  1\n" +
     */
    /*@Test
    fun summary() {
        correctTestModel.use {
            assertEquals("sequential_model", it.name)

            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            assertEquals(
                ModelSummary(
                    type = "Sequential",
                    name = "sequential_model",
                    layersSummaries = listOf(
                        LayerSummary("input_1", "Input", TensorShape(-1, 28, 28, 1), 0, emptyList()),
                        LayerSummary("conv2d_1", "Conv2D", TensorShape(-1, 28, 28, 32), 832, emptyList()),
                        LayerSummary("maxPool_1", "MaxPool2D", TensorShape(-1, 14, 14, 32), 0, emptyList()),
                        LayerSummary("conv2d_2", "Conv2D", TensorShape(-1, 14, 14, 64), 51264, emptyList()),
                        LayerSummary("maxPool_2", "MaxPool2D", TensorShape(-1, 7, 7, 64), 0, emptyList()),
                        LayerSummary("flatten_1", "Flatten", TensorShape(-1, 3136), 0, emptyList()),
                        LayerSummary("dense_1", "Dense", TensorShape(-1, 512), 1606144, emptyList()),
                        LayerSummary("dense_2", "Dense", TensorShape(-1, 10), 5130, emptyList())
                    ),
                    trainableParamsCount = 1663370,
                    frozenParamsCount = 0
                ),
                it.summary()
            )

            assertTrue(
                it.kGraph().toString().contentEquals(
                    "Name: default_data_placeholder; Type: Placeholder; Out #tensors:  1\n" +
                            "Name: conv2d_1_conv2d_kernel; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: conv2d_1_conv2d_bias; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Const; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_1; Type: Const; Out #tensors:  1\n" +
                            "Name: StatelessTruncatedNormal; Type: StatelessTruncatedNormal; Out #tensors:  1\n" +
                            "Name: Const_2; Type: Const; Out #tensors:  1\n" +
                            "Name: Cast; Type: Cast; Out #tensors:  1\n" +
                            "Name: Init_conv2d_1_conv2d_kernel; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign_conv2d_1_conv2d_kernel; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_3; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_conv2d_1_conv2d_bias/Zero; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_conv2d_1_conv2d_bias/Fill; Type: Fill; Out #tensors:  1\n" +
                            "Name: Assign_conv2d_1_conv2d_bias; Type: Assign; Out #tensors:  1\n" +
                            "Name: conv2d_2_conv2d_kernel; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: conv2d_2_conv2d_bias; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Const_4; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_5; Type: Const; Out #tensors:  1\n" +
                            "Name: StatelessTruncatedNormal_1; Type: StatelessTruncatedNormal; Out #tensors:  1\n" +
                            "Name: Const_6; Type: Const; Out #tensors:  1\n" +
                            "Name: Cast_1; Type: Cast; Out #tensors:  1\n" +
                            "Name: Init_conv2d_2_conv2d_kernel; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign_conv2d_2_conv2d_kernel; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_7; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_conv2d_2_conv2d_bias/Zero; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_conv2d_2_conv2d_bias/Fill; Type: Fill; Out #tensors:  1\n" +
                            "Name: Assign_conv2d_2_conv2d_bias; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_8; Type: Const; Out #tensors:  1\n" +
                            "Name: dense_1_dense_kernel; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: dense_1_dense_bias; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Const_9; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_10; Type: Const; Out #tensors:  1\n" +
                            "Name: StatelessTruncatedNormal_2; Type: StatelessTruncatedNormal; Out #tensors:  1\n" +
                            "Name: Const_11; Type: Const; Out #tensors:  1\n" +
                            "Name: Cast_2; Type: Cast; Out #tensors:  1\n" +
                            "Name: Init_dense_1_dense_kernel; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign_dense_1_dense_kernel; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_12; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_13; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_dense_1_dense_bias; Type: Fill; Out #tensors:  1\n" +
                            "Name: Assign_dense_1_dense_bias; Type: Assign; Out #tensors:  1\n" +
                            "Name: dense_2_dense_kernel; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: dense_2_dense_bias; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Const_14; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_15; Type: Const; Out #tensors:  1\n" +
                            "Name: StatelessTruncatedNormal_3; Type: StatelessTruncatedNormal; Out #tensors:  1\n" +
                            "Name: Const_16; Type: Const; Out #tensors:  1\n" +
                            "Name: Cast_3; Type: Cast; Out #tensors:  1\n" +
                            "Name: Init_dense_2_dense_kernel; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign_dense_2_dense_kernel; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_17; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_18; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_dense_2_dense_bias; Type: Fill; Out #tensors:  1\n" +
                            "Name: Assign_dense_2_dense_bias; Type: Assign; Out #tensors:  1\n" +
                            "Name: Placeholder; Type: Placeholder; Out #tensors:  1\n" +
                            "Name: numberOfLosses; Type: Placeholder; Out #tensors:  1\n" +
                            "Name: training; Type: Placeholder; Out #tensors:  1\n" +
                            "Name: Conv2d; Type: Conv2D; Out #tensors:  1\n" +
                            "Name: BiasAdd; Type: BiasAdd; Out #tensors:  1\n" +
                            "Name: Relu; Type: Relu; Out #tensors:  1\n" +
                            "Name: Activation_conv2d_1; Type: Identity; Out #tensors:  1\n" +
                            "Name: Const_19; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_20; Type: Const; Out #tensors:  1\n" +
                            "Name: MaxPool; Type: MaxPoolV2; Out #tensors:  1\n" +
                            "Name: Conv2d_1; Type: Conv2D; Out #tensors:  1\n" +
                            "Name: BiasAdd_1; Type: BiasAdd; Out #tensors:  1\n" +
                            "Name: Relu_1; Type: Relu; Out #tensors:  1\n" +
                            "Name: Activation_conv2d_2; Type: Identity; Out #tensors:  1\n" +
                            "Name: Const_21; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_22; Type: Const; Out #tensors:  1\n" +
                            "Name: MaxPool_1; Type: MaxPoolV2; Out #tensors:  1\n" +
                            "Name: Reshape; Type: Reshape; Out #tensors:  1\n" +
                            "Name: MatMul; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Add; Type: Add; Out #tensors:  1\n" +
                            "Name: Relu_2; Type: Relu; Out #tensors:  1\n" +
                            "Name: Activation_dense_1; Type: Identity; Out #tensors:  1\n" +
                            "Name: MatMul_1; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Add_1; Type: Add; Out #tensors:  1\n" +
                            "Name: Activation_dense_2; Type: Identity; Out #tensors:  1\n" +
                            "Name: SoftmaxCrossEntropyWithLogits; Type: SoftmaxCrossEntropyWithLogits; Out #tensors:  2\n" +
                            "Name: Const_23; Type: Const; Out #tensors:  1\n" +
                            "Name: Mean; Type: Mean; Out #tensors:  1\n" +
                            "Name: default_training_loss; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/OnesLike; Type: OnesLike; Out #tensors:  1\n" +
                            "Name: Gradients/Identity; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Shape; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Const; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Const_1; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Size; Type: Size; Out #tensors:  1\n" +
                            "Name: Gradients/Add; Type: Add; Out #tensors:  1\n" +
                            "Name: Gradients/Mod; Type: Mod; Out #tensors:  1\n" +
                            "Name: Gradients/Range; Type: Range; Out #tensors:  1\n" +
                            "Name: Gradients/OnesLike_1; Type: OnesLike; Out #tensors:  1\n" +
                            "Name: Gradients/DynamicStitch; Type: DynamicStitch; Out #tensors:  1\n" +
                            "Name: Gradients/Const_2; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Maximum; Type: Maximum; Out #tensors:  1\n" +
                            "Name: Gradients/Div; Type: Div; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/Tile; Type: Tile; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_1; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_2; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Const_3; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Prod; Type: Prod; Out #tensors:  1\n" +
                            "Name: Gradients/Prod_1; Type: Prod; Out #tensors:  1\n" +
                            "Name: Gradients/Const_4; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Maximum_1; Type: Maximum; Out #tensors:  1\n" +
                            "Name: Gradients/Div_1; Type: Div; Out #tensors:  1\n" +
                            "Name: Gradients/Cast; Type: Cast; Out #tensors:  1\n" +
                            "Name: Gradients/Div_2; Type: Div; Out #tensors:  1\n" +
                            "Name: Gradients/ZerosLike; Type: ZerosLike; Out #tensors:  1\n" +
                            "Name: Gradients/Const_5/Const; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/ExpandDims; Type: ExpandDims; Out #tensors:  1\n" +
                            "Name: Gradients/Multiply; Type: Mul; Out #tensors:  1\n" +
                            "Name: Gradients/LogSoftmax; Type: LogSoftmax; Out #tensors:  1\n" +
                            "Name: Gradients/Const_6/Const; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/Multiply_1; Type: Mul; Out #tensors:  1\n" +
                            "Name: Gradients/Const_7/Const; Type: Const; Out #tensors:  1\n" +
                            "Name: Gradients/ExpandDims_1; Type: ExpandDims; Out #tensors:  1\n" +
                            "Name: Gradients/Multiply_2; Type: Mul; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_1; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_2; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_3; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_3; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_4; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/BroadcastGradientArgs; Type: BroadcastGradientArgs; Out #tensors:  2\n" +
                            "Name: Gradients/Sum; Type: Sum; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape_1; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/Sum_1; Type: Sum; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape_2; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/MatMul; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Gradients/MatMul_1; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_4; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/ReluGrad; Type: ReluGrad; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_5; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_6; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_5; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_6; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/BroadcastGradientArgs_1; Type: BroadcastGradientArgs; Out #tensors:  2\n" +
                            "Name: Gradients/Sum_2; Type: Sum; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape_3; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/Sum_3; Type: Sum; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape_4; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/MatMul_2; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Gradients/MatMul_3; Type: MatMul; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_7; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Reshape_5; Type: Reshape; Out #tensors:  1\n" +
                            "Name: Gradients/MaxPoolGradV2; Type: MaxPoolGradV2; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_7; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/ReluGrad_1; Type: ReluGrad; Out #tensors:  1\n" +
                            "Name: Gradients/BiasAddGrad; Type: BiasAddGrad; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_8; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_8; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Conv2DBackpropInput; Type: Conv2DBackpropInput; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_9; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Conv2DBackpropFilter; Type: Conv2DBackpropFilter; Out #tensors:  1\n" +
                            "Name: Gradients/MaxPoolGradV2_1; Type: MaxPoolGradV2; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_9; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/ReluGrad_2; Type: ReluGrad; Out #tensors:  1\n" +
                            "Name: Gradients/BiasAddGrad_1; Type: BiasAddGrad; Out #tensors:  1\n" +
                            "Name: Gradients/Identity_10; Type: Identity; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_10; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Conv2DBackpropInput_1; Type: Conv2DBackpropInput; Out #tensors:  1\n" +
                            "Name: Gradients/Shape_11; Type: Shape; Out #tensors:  1\n" +
                            "Name: Gradients/Conv2DBackpropFilter_1; Type: Conv2DBackpropFilter; Out #tensors:  1\n" +
                            "Name: Shape; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_24; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_1_conv2d_kernel-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_1_conv2d_kernel-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_1_conv2d_kernel-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_1; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_25; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_1_conv2d_kernel-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_1_conv2d_kernel-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_1_conv2d_kernel-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_2; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_26; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_1_conv2d_bias-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_1_conv2d_bias-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_1_conv2d_bias-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_3; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_27; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_1_conv2d_bias-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_1_conv2d_bias-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_1_conv2d_bias-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_4; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_28; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_2_conv2d_kernel-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_2_conv2d_kernel-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_2_conv2d_kernel-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_5; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_29; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_2_conv2d_kernel-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_2_conv2d_kernel-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_2_conv2d_kernel-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_6; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_30; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_2_conv2d_bias-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_2_conv2d_bias-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_2_conv2d_bias-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_7; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_31; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_conv2d_2_conv2d_bias-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_conv2d_2_conv2d_bias-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_conv2d_2_conv2d_bias-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_8; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_32; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_1_dense_kernel-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_1_dense_kernel-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_1_dense_kernel-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_9; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_33; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_1_dense_kernel-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_1_dense_kernel-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_1_dense_kernel-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_10; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_34; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_1_dense_bias-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_1_dense_bias-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_1_dense_bias-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_11; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_35; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_1_dense_bias-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_1_dense_bias-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_1_dense_bias-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_12; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_36; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_2_dense_kernel-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_2_dense_kernel-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_2_dense_kernel-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_13; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_37; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_2_dense_kernel-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_2_dense_kernel-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_2_dense_kernel-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_14; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_38; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_2_dense_bias-m; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_2_dense_bias-m; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_2_dense_bias-m; Type: Assign; Out #tensors:  1\n" +
                            "Name: Shape_15; Type: Shape; Out #tensors:  1\n" +
                            "Name: Const_39; Type: Const; Out #tensors:  1\n" +
                            "Name: Init_optimizer_dense_2_dense_bias-v; Type: Fill; Out #tensors:  1\n" +
                            "Name: optimizer_dense_2_dense_bias-v; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_dense_2_dense_bias-v; Type: Assign; Out #tensors:  1\n" +
                            "Name: optimizer_beta1_power; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Init_optimizer_beta1_power; Type: Const; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_beta1_power; Type: Assign; Out #tensors:  1\n" +
                            "Name: optimizer_beta2_power; Type: VariableV2; Out #tensors:  1\n" +
                            "Name: Init_optimizer_beta2_power; Type: Const; Out #tensors:  1\n" +
                            "Name: Assign_optimizer_beta2_power; Type: Assign; Out #tensors:  1\n" +
                            "Name: Const_40; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_41; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_42; Type: Const; Out #tensors:  1\n" +
                            "Name: Const_43; Type: Const; Out #tensors:  1\n" +
                            "Name: ApplyAdam; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_1; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_2; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_3; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_4; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_5; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_6; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: ApplyAdam_7; Type: ApplyAdam; Out #tensors:  1\n" +
                            "Name: Mul; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign; Type: Assign; Out #tensors:  1\n" +
                            "Name: Mul_1; Type: Mul; Out #tensors:  1\n" +
                            "Name: Assign_1; Type: Assign; Out #tensors:  1\n" +
                            "Name: default_output; Type: Softmax; Out #tensors:  1\n" +
                            "Name: Const_44; Type: Const; Out #tensors:  1\n" +
                            "Name: ArgMax; Type: ArgMax; Out #tensors:  1\n" +
                            "Name: Const_45; Type: Const; Out #tensors:  1\n" +
                            "Name: ArgMax_1; Type: ArgMax; Out #tensors:  1\n" +
                            "Name: Equal; Type: Equal; Out #tensors:  1\n" +
                            "Name: Cast_4; Type: Cast; Out #tensors:  1\n" +
                            "Name: Const_46; Type: Const; Out #tensors:  1\n" +
                            "Name: Mean_1; Type: Mean; Out #tensors:  1\n"
                )
            )
        }
    }*/

    @Test
    fun compilation() {
        val correctTestModel = Sequential.of(correctTestModelLayers).apply {
            name = "sequential_model"
        }

        val exception =
            assertThrows(UninitializedPropertyAccessException::class.java) { correctTestModel.layers[1].paramCount }
        assertEquals(
            "lateinit property kernel has not been initialized",
            exception.message
        )

        assertFalse(correctTestModel.isModelCompiled)

        correctTestModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())
            assertTrue(correctTestModel.isModelCompiled)

            assertEquals(it.layers[0].paramCount, 0)
            assertEquals(it.layers[1].paramCount, 832)
            assertEquals(it.layers[2].paramCount, 0)
            assertEquals(it.layers[3].paramCount, 51264)
            assertEquals(it.layers[4].paramCount, 0)
            assertEquals(it.layers[5].paramCount, 0)
            assertEquals(it.layers[6].paramCount, 1606144)
            assertEquals(it.layers[7].paramCount, 5130)

            assertArrayEquals(it.layers[0].outputShape.dims(), longArrayOf(-1, 28, 28, 1))
            assertArrayEquals(it.layers[1].outputShape.dims(), longArrayOf(-1, 28, 28, 32))
            assertArrayEquals(it.layers[2].outputShape.dims(), longArrayOf(-1, 14, 14, 32))
            assertArrayEquals(it.layers[3].outputShape.dims(), longArrayOf(-1, 14, 14, 64))
            assertArrayEquals(it.layers[4].outputShape.dims(), longArrayOf(-1, 7, 7, 64))
            assertArrayEquals(it.layers[5].outputShape.dims(), longArrayOf(-1, 3136))
            assertArrayEquals(it.layers[6].outputShape.dims(), longArrayOf(-1, 512))
            assertArrayEquals(it.layers[7].outputShape.dims(), longArrayOf(-1, 10))
        }
    }

    @Test
    fun compilationWithTwoMetrics() {
        val correctTestModel = Sequential.of(correctTestModelLayers).apply {
            name = "sequential_model"
        }

        val exception =
            assertThrows(UninitializedPropertyAccessException::class.java) { correctTestModel.layers[1].paramCount }
        assertEquals(
            "lateinit property kernel has not been initialized",
            exception.message
        )

        assertFalse(correctTestModel.isModelCompiled)

        correctTestModel.use {
            it.compile(
                optimizer = Adam(),
                loss = SoftmaxCrossEntropyWithLogits(),
                metrics = listOf(Accuracy(), Accuracy())
            )
            assertTrue(correctTestModel.isModelCompiled)

            assertEquals(it.layers[0].paramCount, 0)
            assertEquals(it.layers[1].paramCount, 832)
            assertEquals(it.layers[2].paramCount, 0)
            assertEquals(it.layers[3].paramCount, 51264)
            assertEquals(it.layers[4].paramCount, 0)
            assertEquals(it.layers[5].paramCount, 0)
            assertEquals(it.layers[6].paramCount, 1606144)
            assertEquals(it.layers[7].paramCount, 5130)

            assertArrayEquals(it.layers[0].outputShape.dims(), longArrayOf(-1, 28, 28, 1))
            assertArrayEquals(it.layers[1].outputShape.dims(), longArrayOf(-1, 28, 28, 32))
            assertArrayEquals(it.layers[2].outputShape.dims(), longArrayOf(-1, 14, 14, 32))
            assertArrayEquals(it.layers[3].outputShape.dims(), longArrayOf(-1, 14, 14, 64))
            assertArrayEquals(it.layers[4].outputShape.dims(), longArrayOf(-1, 7, 7, 64))
            assertArrayEquals(it.layers[5].outputShape.dims(), longArrayOf(-1, 3136))
            assertArrayEquals(it.layers[6].outputShape.dims(), longArrayOf(-1, 512))
            assertArrayEquals(it.layers[7].outputShape.dims(), longArrayOf(-1, 10))
        }
    }

    @Test
    fun repeatableNamesFails() {
        val exception = assertThrows(RepeatableLayerNameException::class.java) {
            Sequential.of(
                Input(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS
                ),
                Conv2D(
                    filters = 32,
                    kernelSize = intArrayOf(5, 5),
                    strides = intArrayOf(1, 1, 1, 1),
                    activation = Activations.Relu,
                    kernelInitializer = HeNormal(SEED),
                    biasInitializer = Zeros(),
                    padding = ConvPadding.SAME,
                    name = "conv2d_1"
                ),
                Conv2D(
                    filters = 64,
                    kernelSize = intArrayOf(5, 5),
                    strides = intArrayOf(1, 1, 1, 1),
                    activation = Activations.Relu,
                    kernelInitializer = HeNormal(SEED),
                    biasInitializer = Zeros(),
                    padding = ConvPadding.SAME,
                    name = "conv2d_1"
                )
            )
        }

        assertEquals(
            "The layer name conv2d_1 is used in previous layers. The layer name should be unique.",
            exception.message
        )
    }

    @Test
    fun failOnIncorrectShape() {
        val heNormal = HeNormal(SEED)

        val vgg11 = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
            ),
            Conv2D(
                filters = 64,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
            ),
            Conv2D(
                filters = 128,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            Conv2D(
                filters = 128,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
            ),
            Conv2D(
                filters = 256,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            Conv2D(
                filters = 256,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
            ),
            Conv2D(
                filters = 128,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            Conv2D(
                filters = 128,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal,
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
            ),
            Flatten(),
            Dense(
                outputSize = 2048,
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal
            ),
            Dense(
                outputSize = 1000,
                activation = Activations.Relu,
                kernelInitializer = heNormal,
                biasInitializer = heNormal
            ),
            Dense(
                outputSize = 10,
                activation = Activations.Linear,
                kernelInitializer = heNormal,
                biasInitializer = heNormal
            )
        )

        vgg11.use {
            assertThrows(IllegalArgumentException::class.java) {
                try {
                    it.compile(
                        optimizer = Adam(),
                        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                        metric = Metrics.ACCURACY
                    )
                } catch (e: Throwable) {
                    throw e
                }
            }
        }
    }

    @Test
    fun namesGeneration() {
        val model = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = intArrayOf(5, 5),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1)
            ),
            Flatten(),
            Dense(
                outputSize = AMOUNT_OF_CLASSES,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f)
            )
        )

        assertEquals(model.layers[1].name, "conv2d_2")
        assertEquals(model.layers[2].name, "maxpool2d_3")
        assertEquals(model.layers[3].name, "flatten_4")
        assertEquals(model.layers[4].name, "dense_5")
    }
}
