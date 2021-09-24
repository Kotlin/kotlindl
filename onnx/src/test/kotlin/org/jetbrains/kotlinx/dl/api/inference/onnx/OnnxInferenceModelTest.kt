package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.io.File
import kotlin.math.roundToInt

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class OnnxInferenceModelTest {

    val modelType = ONNXModels.CV.ResNet18
    lateinit var model: OnnxInferenceModel

    @BeforeEach
    fun setup() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        model = modelHub.loadModel(modelType)
    }

    @Test
    fun loadModel() {
        model.use {
            assertNotNull(it)
        }
    }

    @Test
    fun checkModel() {
        model.use {
            assertEquals(4, model.inputShape.size)
            assertEquals(3, model.outputShape.size)
        }
    }

    @Test
    fun basicPrediction() {
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

                val res = it.predict(inputData)
                println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

                val top5 = predictTopNLabels(it, inputData, imageNetClassLabels)

                //None of the predictions were less than 5 for the given image inputs
                //Look for any predictions less than 5
                val predictions = top5.map {
                    it.value.second < 5f
                }

                println(top5.toString())

                assertNotNull(top5)
                assertEquals(5, top5.size)

                //Verify there are no predictions less than 5
                assertEquals(false, predictions.contains(true))
            }
        }
    }

    @Test
    fun softMaxPrediction() {
        val imageNetClassLabels = loadImageNetClassLabels()
        var totalProbability = 0f
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

                val res = it.predictSoftly(inputData)

                //Verify the array has 1000 probabilities
                assertEquals(1000, res.size)

                //Verify the 1000 array probabilities add up to 1
                var newPorbability = 0f

                for (prob in res) newPorbability += prob

                totalProbability += newPorbability
            }
        }
        assertEquals(1, totalProbability.roundToInt())
    }
}