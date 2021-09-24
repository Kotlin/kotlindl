package org.jetbrains.kotlinx.dl.api.inference.loaders

import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File

class ONNXModelHubTest {
    @Test
    fun basicLoadTest(@TempDir tempDir: File) {
        val onnxModel = tempDir.resolve(ONNXModels.CV.ResNet18.modelRelativePath + ".onnx")
        ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet18)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertEquals(onnxModel.isFile, true)
    }
}