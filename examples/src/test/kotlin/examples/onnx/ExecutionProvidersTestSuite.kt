package examples.onnx

import examples.onnx.cv.runImageRecognitionPrediction
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProviders.ExecutionProvider.CUDA
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Disabled

class ExecutionProvidersTestSuite {
    private fun resnetModelsInference(executionProvider: ExecutionProvider) {
        val modelsToTest = listOf(
            ONNXModels.CV.ResNet101(),
            ONNXModels.CV.ResNet101v2(),
            ONNXModels.CV.ResNet152(),
            ONNXModels.CV.ResNet152v2(),
            ONNXModels.CV.ResNet18(),
            ONNXModels.CV.ResNet18v2(),
            ONNXModels.CV.ResNet34(),
            ONNXModels.CV.ResNet34v2(),
            ONNXModels.CV.ResNet50(),
            ONNXModels.CV.ResNet50v2(),
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
}
