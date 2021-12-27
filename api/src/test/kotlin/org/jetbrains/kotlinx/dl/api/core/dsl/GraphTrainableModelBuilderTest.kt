package org.jetbrains.kotlinx.dl.api.core.dsl

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
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
                    +Conv1D()
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
                    +Conv1D()(+Input(128, 128))
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
    fun genericBuilder() {
        assert(
            ::Functional {
                layers {
                    +Input(128, 128, name = "First Layer")
                    +Conv1D(name = "Second Layer")
                }
            }.let {
                it.layers.size == 2 && it.layers[0] is Input && it.layers[1] is Conv1D
            }
        )
    }
}