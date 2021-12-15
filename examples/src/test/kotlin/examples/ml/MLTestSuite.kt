/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.ml

import org.jetbrains.kotlinx.dl.logging.api.GlobalLogFactory
import org.jetbrains.kotlinx.dl.logging.core.DefaultLogFactory
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class MLTestSuite {
    @BeforeEach
    fun initLogger() {
        GlobalLogFactory = DefaultLogFactory.also {
            it.setup()
        }
    }


    @Test
    fun irisClassificationTest() {
        irisClassification()
    }

    @Test
    fun linearRegressionTest() {
        linearRegression()
    }

    @Test
    fun sineRegressionTest() {
        sineRegression()
    }
}
