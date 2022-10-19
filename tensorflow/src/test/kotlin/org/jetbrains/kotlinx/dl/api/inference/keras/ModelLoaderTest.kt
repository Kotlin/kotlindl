/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File

private const val INITIALIZER_PATH = "src/test/resources/inference/keras/ModelLoader/initializer_identity.json"

class ModelLoaderTest {
    @Test
    fun loadInitializersFromFile() {
        val initializerFile = File(INITIALIZER_PATH)

        val modelConfiguration = loadSequentialModelConfiguration(initializerFile)

        assertEquals(14, modelConfiguration.layers.size)
        with(modelConfiguration.layers) {
            with((this[1] as Dense).kernelInitializer) {
                assertTrue(this is GlorotUniform)
                assertEquals(1.0, (this as GlorotUniform).scale)
            }

            with((this[2] as Dense).kernelInitializer) {
                assertTrue(this is GlorotNormal)
                assertEquals(1.0, (this as GlorotNormal).scale)
            }

            with((this[3] as Dense).kernelInitializer) {
                assertTrue(this is HeNormal)
                assertEquals(2.0, (this as HeNormal).scale)
            }

            with((this[4] as Dense).kernelInitializer) {
                assertTrue(this is HeUniform)
                assertEquals(2.0, (this as HeUniform).scale)
            }

            with((this[5] as Dense).kernelInitializer) {
                assertTrue(this is LeCunNormal)
                assertEquals(1.0, (this as LeCunNormal).scale)
            }

            with((this[6] as Dense).kernelInitializer) {
                assertTrue(this is LeCunUniform)
                assertEquals(1.0, (this as LeCunUniform).scale)
            }

            with((this[7] as Dense).kernelInitializer) {
                assertTrue(this is Zeros)
            }

            with((this[8] as Dense).kernelInitializer) {
                assertTrue(this is Constant)
                assertEquals(2f, (this as Constant).constantValue)
            }

            with((this[9] as Dense).kernelInitializer) {
                assertTrue(this is Ones)
            }

            with((this[10] as Dense).kernelInitializer) {
                assertTrue(this is RandomNormal)
                assertEquals(0.0f, (this as RandomNormal).mean)
                assertEquals(1.0f, this.stdev)
            }

            with((this[11] as Dense).kernelInitializer) {
                assertTrue(this is RandomUniform)
                assertEquals(0.05f, (this as RandomUniform).maxVal)
                assertEquals(-0.05f, this.minVal)
            }

            with((this[12] as Dense).kernelInitializer) {
                assertTrue(this is TruncatedNormal)
            }

            with((this[13] as Dense).kernelInitializer) {
                assertTrue(this is Identity)
                assertEquals(3.4f, (this as Identity).gain)
            }
        }
    }
}
