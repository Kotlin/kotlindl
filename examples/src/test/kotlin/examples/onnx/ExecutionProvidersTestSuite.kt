package examples.onnx

import examples.onnx.cv.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CUDA
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.assertThrows
import java.io.File

class ExecutionProvidersTestSuite {
    private fun resnetModelsInference(executionProvider: ExecutionProvider) {
        val modelsToTest = listOf(
            ONNXModels.CV.ResNet101,
            ONNXModels.CV.ResNet101v2,
            ONNXModels.CV.ResNet152,
            ONNXModels.CV.ResNet152v2,
            ONNXModels.CV.ResNet18,
            ONNXModels.CV.ResNet18v2,
            ONNXModels.CV.ResNet34,
            ONNXModels.CV.ResNet34v2,
            ONNXModels.CV.ResNet50,
            ONNXModels.CV.ResNet50v2,
            ONNXModels.CV.ResNet50custom,
        )

        assertDoesNotThrow {
            for (modelType in modelsToTest) {
                val defaultExecutorResults = runImageRecognitionPrediction(modelType)
                val differentExecutorResults = runImageRecognitionPrediction(modelType, listOf(executionProvider))

                assertEquals(defaultExecutorResults.map { it.first }, differentExecutorResults.map { it.first })
            }
        }
    }

    @Test
    fun defaultCpuTest() {
        resnetModelsInference(CPU())
    }

    @Test
    fun cpuArenaAllocatorDisabledTest() {
        resnetModelsInference(CPU(false))
    }

    @Test
    @Disabled("CUDA environment should be set up for this test")
    fun cudaTest() {
        resnetModelsInference(CUDA())
    }

    @Test
    fun executorProvidersComparisonTest() {
        assertEquals(CPU(), CPU(true))

        assertNotEquals(CPU(), CPU(false))

        assertNotEquals(CUDA(), CUDA(1))

        assertEquals(
            listOf(CPU(), CUDA()),
            listOf(CPU(), CUDA())
        )

        assertNotEquals(
            listOf(CPU(), CUDA()),
            listOf(CPU(false), CUDA())
        )
    }

    @Test
    fun executionProvidersDuplicatesTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = modelHub.loadModel(ONNXModels.CV.ResNet18)

        model.use {
            assertDoesNotThrow {
                model.initializeWith(CPU(), CPU(), CPU())
            }
        }
    }

    @Test
    fun twoCpuExecutorsWithDifferentAllocatorsTest() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val model = modelHub.loadModel(ONNXModels.CV.ResNet18)

        model.use {
            assertThrows<IllegalArgumentException> {
                model.initializeWith(CPU(), CPU(false))
            }
        }
    }
}
