/*
 * Copyright 2022-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.assertThrows
import kotlin.random.Random

class OnnxOutputsSupportTestSuite {
    private val pathToModel: String = getFileFromResource("models/onnx/lgbmSequenceOutput.onnx").absolutePath
    private val model = OnnxInferenceModel.load(pathToModel)
    private val features = (1..27).map { Random.nextFloat() }.toFloatArray()

    @Test
    fun predictSoftlyLgbmSequenceOutputTest() {
        assertThrows<IllegalArgumentException> {
            model.predictSoftly(features to TensorShape(27), "probabilities")
        }
    }

    @Test
    fun predictRawLgbmSequenceOutputTest() {
        assertDoesNotThrow {
            model.predictRaw(features to TensorShape(27))
        }
    }
}
