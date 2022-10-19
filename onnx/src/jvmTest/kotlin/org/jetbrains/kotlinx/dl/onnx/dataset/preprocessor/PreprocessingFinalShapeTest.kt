/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.awt.image.BufferedImage

class PreprocessingFinalShapeTest {
    @Test
    fun transposeOutputShapeTest() {
        val preprocess = pipeline<BufferedImage>()
            .toFloatArray { }
            .transpose { axes = intArrayOf(1, 2, 0) }

        val image = BufferedImage(10, 20, BufferedImage.TYPE_3BYTE_BGR)
        val (_, actualShape) = preprocess.apply(image)

        assertEquals(actualShape, preprocess.getOutputShape(TensorShape(10, 20, 3)))
    }

    @Test
    fun transposeInvalidAxesTest() {
        val preprocess = pipeline<BufferedImage>()
            .toFloatArray { }
            .transpose { axes = intArrayOf(1, 2) } // invalid axes

        val image = BufferedImage(10, 20, BufferedImage.TYPE_3BYTE_BGR)

        assertThrows<IllegalArgumentException> { preprocess.apply(image) }
    }
}
