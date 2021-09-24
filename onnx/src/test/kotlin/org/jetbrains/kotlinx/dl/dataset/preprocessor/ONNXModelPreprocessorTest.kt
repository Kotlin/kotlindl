package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.getFileFromResource
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import java.io.File

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ONNXModelPreprocessorTest {
    val modelType = ONNXModels.CV.ResNet50
    lateinit var model: OnnxInferenceModel

    @BeforeEach
    fun setup() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        model = modelHub.loadModel(modelType)
    }

    @Test
    fun preprocessInputs() {
        val imageNetClassLabels = loadImageNetClassLabels()
        model.use {
            println(it)

            for (i in 1..8) {
                val preprocessing: Preprocessing = preprocess {
                    transformImage {
                        load {
                            pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                            imageShape = ImageShape(224, 224, 3)
                            colorMode = ColorOrder.BGR
                        }
                    }
                }

                val inputData = modelType.preprocessInput(preprocessing)

                val onnxModelPreprocessor = ONNXModelPreprocessor(onnxModel = it)
                assertNotNull(onnxModelPreprocessor)
                val predict = onnxModelPreprocessor.apply(inputData, ImageShape(224, 224, 3))

                //The prediction Array should be 1 dimensional with a size of 1000
                assertEquals(1000, predict.size)
            }
        }
    }
}