/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class MultipleInputTestSuite {
    private val pathToModel: String = getFileFromResource("models/onnx/addnet.onnx").absolutePath

    @Test
    fun testCorrectInputs() {
        OnnxInferenceModel.load(pathToModel).use { model ->
            val x = floatArrayOf(1f, 2f, 3f) to TensorShape(3)
            val y = floatArrayOf(1f, 1f, 1f) to TensorShape(3)
            val result = model.predictRaw(mapOf("X" to x, "Y" to y)) { output ->
                return@predictRaw output.getFloatArray("Z")
            }
            Assertions.assertArrayEquals(floatArrayOf(2f, 3f, 4f), result)
        }
    }

    @Test
    fun testIncorrectInputs() {
        OnnxInferenceModel.load(pathToModel).use { model ->
            val x = floatArrayOf(1f, 2f, 3f) to TensorShape(4)
            val y = floatArrayOf(1f, 1f, 1f) to TensorShape(3)
            assertThrows<IllegalArgumentException> {
                model.predictRaw(mapOf("X" to x, "Y" to y)) {}
            }
            assertThrows<IllegalArgumentException> {
                model.predictRaw(mapOf("Xxx" to y, "Y" to y)) {}
            }
        }
    }
}