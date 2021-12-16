package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.dsl.functional
import org.jetbrains.kotlinx.dl.api.core.dsl.sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.io.File

class InputLayerPersistenceTest {
    private lateinit var tempFile: File

    @BeforeEach
    fun createTempFile() {
        tempFile = File.createTempFile("model", ".json")
    }

    @AfterEach
    fun deleteTempFile() {
        tempFile.delete()
    }

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

    private fun testSequentialModel(originalModel: Sequential) {
        originalModel.saveModelConfiguration(tempFile)
        val restoredModel = Sequential.loadModelConfiguration(tempFile)
        assertTrue(originalModel.inputDimensions.contentEquals(restoredModel.inputDimensions))
    }

    private fun testFunctionalModel(originalModel: Functional) {
        originalModel.saveModelConfiguration(tempFile)
        val restoredModel = Functional.loadModelConfiguration(tempFile)
        assertTrue(originalModel.inputDimensions.contentEquals(restoredModel.inputDimensions))
    }
}
