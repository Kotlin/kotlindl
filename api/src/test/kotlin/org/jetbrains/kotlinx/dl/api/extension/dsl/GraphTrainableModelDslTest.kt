package org.jetbrains.kotlinx.dl.api.extension.dsl

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class GraphTrainableModelDslTest {
    @Test
    fun buildSequential() {
        assertDoesNotThrow {
            sequential {
                model {
                    name = "Test Sequential"
                }

                layers {
                    +Input(128, 128)
                    +Conv1D()
                }
            }
        }

    }

    @Test
    fun buildFunctional() {
        assertDoesNotThrow {
            functional {
                model {
                    name = "Test Functional"
                }

                layers {
                    +Input(128, 128)
                    +Conv1D()
                }
            }
        }
    }

    @Test
    fun genericBuilder() {
        assertDoesNotThrow {
            ::Functional {
                layers {
                    +Input(128, 128, name = "First Layer")
                    +Conv1D(name = "Second Layer")
                }
            }
        }
    }
}