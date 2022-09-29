/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Ones
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding2D
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val BATCH_SIZE = 1
private const val NUM_CHANNELS = 1
private const val IMAGE_SIZE = 3

internal class ZeroPadding2DTest {
    @Test
    fun oneArgumentChannelsLast() {
        val padding = 1
        val inputDimensionsArray = intArrayOf(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        val expectedOutputSize = IMAGE_SIZE + 2 * padding

        EagerSession.create().use {
            val tf = Ops.create(it)
            val paddingLayer = ZeroPadding2D(padding, dataFormat = CHANNELS_LAST)
            val inputDimensions = tf.constant(inputDimensionsArray)
            val input = Ones().initialize(1, 1, tf, inputDimensions, "test_input")
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = paddingLayer.build(tf, input, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                BATCH_SIZE.toLong(),
                expectedOutputSize.toLong(),
                expectedOutputSize.toLong(),
                NUM_CHANNELS.toLong()
            )
            val actualShape = shapeFromDims(*output.shape())
            assertEquals(expectedShape, actualShape)

            val actualArray = Array(BATCH_SIZE) {
                Array(expectedOutputSize) {
                    Array(expectedOutputSize) {
                        FloatArray(NUM_CHANNELS) { 0f }
                    }
                }
            }
            output.copyTo(actualArray)

            for (batch in 0 until BATCH_SIZE) {
                for (i in 0 until expectedOutputSize) {
                    for (j in 0 until expectedOutputSize) {
                        for (channel in 0 until NUM_CHANNELS) {
                            if ((i < padding || i >= IMAGE_SIZE + padding) ||
                                (j < padding || j >= IMAGE_SIZE + padding)
                            ) {
                                assertEquals(0f, actualArray[batch][i][j][channel])
                            } else {
                                assertEquals(1f, actualArray[batch][i][j][channel])
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun twoArgumentsChannelsLast() {
        val paddingHeight = 1
        val paddingWidth = 2
        val paddingArray = paddingHeight to paddingWidth
        val inputDimensionsArray = intArrayOf(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        val expectedOutputHeight = IMAGE_SIZE + paddingHeight * 2
        val expectedOutputWidth = IMAGE_SIZE + paddingWidth * 2

        EagerSession.create().use {
            val tf = Ops.create(it)
            val paddingLayer = ZeroPadding2D(paddingArray, dataFormat = CHANNELS_LAST)
            val inputDimensions = tf.constant(inputDimensionsArray)
            val input = Ones().initialize(1, 1, tf, inputDimensions, "test_input")
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = paddingLayer.build(tf, input, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                BATCH_SIZE.toLong(),
                expectedOutputHeight.toLong(),
                expectedOutputWidth.toLong(),
                NUM_CHANNELS.toLong()
            )
            val actualShape = shapeFromDims(*output.shape())
            assertEquals(expectedShape, actualShape)

            val actualArray = Array(BATCH_SIZE) {
                Array(expectedOutputHeight) {
                    Array(expectedOutputWidth) {
                        FloatArray(NUM_CHANNELS) { 0f }
                    }
                }
            }
            output.copyTo(actualArray)

            for (batch in 0 until BATCH_SIZE) {
                for (i in 0 until expectedOutputHeight) {
                    for (j in 0 until expectedOutputWidth) {
                        for (channel in 0 until NUM_CHANNELS) {
                            if ((i < paddingHeight || i >= IMAGE_SIZE + paddingHeight) ||
                                (j < paddingWidth || j >= IMAGE_SIZE + paddingWidth)
                            ) {
                                assertEquals(0f, actualArray[batch][i][j][channel])
                            } else {
                                assertEquals(1f, actualArray[batch][i][j][channel])
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun fourArgumentsChannelsLast() {
        val paddingTop = 1
        val paddingBottom = 3
        val paddingLeft = 2
        val paddingRight = 4
        val paddingArray = intArrayOf(paddingTop, paddingBottom, paddingLeft, paddingRight)
        val inputDimensionsArray = intArrayOf(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        val expectedOutputHeight = IMAGE_SIZE + paddingTop + paddingBottom
        val expectedOutputWidth = IMAGE_SIZE + paddingLeft + paddingRight

        EagerSession.create().use {
            val tf = Ops.create(it)
            val paddingLayer = ZeroPadding2D(paddingArray, dataFormat = CHANNELS_LAST)
            val inputDimensions = tf.constant(inputDimensionsArray)
            val input = Ones().initialize(1, 1, tf, inputDimensions, "test_input")
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = paddingLayer.build(tf, input, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                BATCH_SIZE.toLong(),
                expectedOutputHeight.toLong(),
                expectedOutputWidth.toLong(),
                NUM_CHANNELS.toLong()
            )
            val actualShape = shapeFromDims(*output.shape())
            assertEquals(expectedShape, actualShape)

            val actualArray = Array(BATCH_SIZE) {
                Array(expectedOutputHeight) {
                    Array(expectedOutputWidth) {
                        FloatArray(NUM_CHANNELS) { 0f }
                    }
                }
            }
            output.copyTo(actualArray)

            for (batch in 0 until BATCH_SIZE) {
                for (i in 0 until expectedOutputHeight) {
                    for (j in 0 until expectedOutputWidth) {
                        for (channel in 0 until NUM_CHANNELS) {
                            if ((i < paddingTop || i >= IMAGE_SIZE + paddingTop) ||
                                (j < paddingLeft || j >= IMAGE_SIZE + paddingLeft)
                            ) {
                                assertEquals(0f, actualArray[batch][i][j][channel])
                            } else {
                                assertEquals(1f, actualArray[batch][i][j][channel])
                            }
                        }
                    }
                }
            }
        }
    }
}
