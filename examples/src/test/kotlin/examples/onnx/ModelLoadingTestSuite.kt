package examples.onnx

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import java.io.File

class ModelLoadingTestSuite {
    @Test
    fun testLoadingModelFromBytes() {
        val lgbmModel: File = getFileFromResource("models/onnx/lgbmSequenceOutput.onnx")
        val bytes = lgbmModel.readBytes()

        assertDoesNotThrow {
            val model = OnnxInferenceModel(bytes)
            model.initializeWith(CPU())
            model.close()
        }
    }
}
