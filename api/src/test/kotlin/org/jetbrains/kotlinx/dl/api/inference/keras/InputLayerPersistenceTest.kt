/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.dsl.functional
import org.jetbrains.kotlinx.dl.api.core.dsl.sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.junit.jupiter.api.Test

class InputLayerPersistenceTest {
    @Test
    fun inputLayerSequential() {
        LayerPersistenceTest.run(Sequential.of(Input(4)))
        LayerPersistenceTest.run(Sequential.of(Input(128, 128)))
        LayerPersistenceTest.run(Sequential.of(Input(128, 128, 3)))
        LayerPersistenceTest.run(Sequential.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun inputLayerFunctional() {
        LayerPersistenceTest.run(Functional.of(Input(10)))
        LayerPersistenceTest.run(Functional.of(Input(128, 128)))
        LayerPersistenceTest.run(Functional.of(Input(128, 128, 3)))
        LayerPersistenceTest.run(Functional.of(Input(10, 10, 10, 10)))
    }

    @Test
    fun dslBuilderSequential() {
        LayerPersistenceTest.run(
            sequential {
                layers {
                    +Input(4)
                }
            }
        )
        LayerPersistenceTest.run(
            sequential {
                layers {
                    +Input(128, 128)
                }
            }
        )
        LayerPersistenceTest.run(
            sequential {
                layers {
                    +Input(128, 128, 3)
                }
            }
        )
        LayerPersistenceTest.run(
            sequential {
                layers {
                    +Input(10, 10, 10, 10)
                }
            }
        )
    }

    @Test
    fun dslBuilderFunctional() {
        LayerPersistenceTest.run(
            functional {
                layers {
                    +Input(4)
                }
            }
        )
        LayerPersistenceTest.run(
            functional {
                layers {
                    +Input(128, 128)
                }
            }
        )
        LayerPersistenceTest.run(
            functional {
                layers {
                    +Input(128, 128, 3)
                }
            }
        )
        LayerPersistenceTest.run(
            functional {
                layers {
                    +Input(10, 10, 10, 10)
                }
            }
        )
    }
}
