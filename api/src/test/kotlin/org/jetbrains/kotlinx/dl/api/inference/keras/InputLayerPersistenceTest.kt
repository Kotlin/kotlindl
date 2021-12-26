/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.dsl.functional
import org.jetbrains.kotlinx.dl.api.core.dsl.sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.inference.keras.ConvTransposePersistenceTest.Companion.testFunctionalModel
import org.jetbrains.kotlinx.dl.api.inference.keras.ConvTransposePersistenceTest.Companion.testSequentialModel
import org.junit.jupiter.api.Test

class InputLayerPersistenceTest {
    @Test
    fun inputLayerSequential() {
        testSequentialModel(Sequential.of(Input(4)))
        testSequentialModel(Sequential.of(Input(128, 128)))
        testSequentialModel(Sequential.of(Input(128, 128, 3)))
        testSequentialModel(Sequential.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun inputLayerFunctional() {
        testFunctionalModel(Functional.of(Input(10)))
        testFunctionalModel(Functional.of(Input(128, 128)))
        testFunctionalModel(Functional.of(Input(128, 128, 3)))
        testFunctionalModel(Functional.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun dslBuilderSequential() {
        testSequentialModel(
            sequential {
                layers {
                    +Input(4)
                }
            }
        )
        testSequentialModel(
            sequential {
                layers {
                    +Input(128, 128)
                }
            }
        )
        testSequentialModel(
            sequential {
                layers {
                    +Input(128, 128, 3)
                }
            }
        )
        testSequentialModel(
            sequential {
                layers {
                    +Input(10, 10, 10, 10)
                }
            }
        )
    }

    @Test
    fun dslBuilderFunctional() {
        testFunctionalModel(
            functional {
                layers {
                    +Input(4)
                }
            }
        )
        testFunctionalModel(
            functional {
                layers {
                    +Input(128, 128)
                }
            }
        )
        testFunctionalModel(
            functional {
                layers {
                    +Input(128, 128, 3)
                }
            }
        )
        testFunctionalModel(
            functional {
                layers {
                    +Input(10, 10, 10, 10)
                }
            }
        )
    }
}
