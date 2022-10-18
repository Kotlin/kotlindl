/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.dsl

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

internal class GraphTrainableModelBuilderTest {
    @Test
    fun buildSequential() {
        assert(
            sequential {
                model {
                    name = "Test Sequential"
                }

                layers {
                    +Input(128, 128)
                    +Conv1D(32, 3, 1, 1)
                }
            }.let {
                it.layers.size == 2 && it.layers[0] is Input && it.layers[1] is Conv1D
            }
        )

        assert(
            sequential {
                model {
                    name = "Test Sequential"
                }

                layers {
                    +Conv1D(32, 3, 1, 1)(+Input(128, 128))
                }
            }.let {
                it.layers.size == 2 && it.layers[0] is Input && it.layers[1] is Conv1D
            }
        )

    }

    @Test
    fun buildFunctional() {
        assert(
            functional {
                model {
                    name = "Test Functional"
                }

                layers {
                    +Input(128, 128)
                }
            }.let {
                it.layers.size == 1 && it.layers[0] is Input
            }
        )
    }

    @Test
    fun buildFunctionalWithUnboundLayers() {
        val exception = Assertions.assertThrows(IllegalStateException::class.java) {
            functional {
                model {
                    name = "Test Functional"
                }

                layers {
                    +Input(128, 128)
                    +Conv1D(32, 3, 1, 1, name = "Second Layer")
                    +Conv1D(32, 3, 1, 1, name = "Third Layer")
                }
            }
        }
        Assertions.assertEquals(
            "The following layers are not reachable from the input: [Second Layer (Conv1D), Third Layer (Conv1D)]",
            exception.message
        )
    }

    @Test
    fun genericBuilder() {
        assert(
            ::Functional {
                layers {
                    +Input(128, 128, name = "First Layer")
                    +Conv1D(32, 3, 1, 1, name = "Second Layer")
                }
            }.let {
                it.layers.size == 2 && it.layers[0] is Input && it.layers[1] is Conv1D
            }
        )
    }
}
