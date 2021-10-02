package examples.onnx.objectdetection.yolov4

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File
import kotlin.math.max
import kotlin.math.min

object Yolov4 {
    fun predict() {
        val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
        val modelType = ONNXModels.ObjectDetection.YOLOv4
        val model = modelHub.loadModel(modelType)
        model.use {
            println(it)
            for (i in 0..9) {
                val preprocessing: Preprocessing = preprocess {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                    transformImage {
                        resize {
                            outputHeight = 416
                            outputWidth = 416
                        }
                    }
                }

                val inputData = modelType.preprocessInput(preprocessing)
                val predict = it.detectObjects(inputData)
                println(predict.toString())
            }
        }
    }

    private fun OnnxInferenceModel.detectObjects(inputData: FloatArray, topK: Int = 5): List<DetectedObject> {
        // Following https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
        val foundObjects = mutableListOf<DetectedObject>()
        val rawPredictions = this.predictRaw(inputData)
        for (colPredictions in rawPredictions) {
            for (rowPrediction in colPredictions) {
                val predictions = rowPrediction as Array<Array<Array<FloatArray>>>
                for (col in predictions) {
                    for (row in col) {
                        for (block in row) {
                            val xCenter = block[0]
                            val yCenter = block[1]
                            val w = block[2]
                            val h = block[3]
                            val conf = block[4]
                            val idx = block.sliceArray(5..84).argmax()
                            if (conf > 0.6) {
                                val element = DetectedObject(
                                    classLabel = cocoCategories[idx + 1] ?: "",
                                    probability = conf,
                                    xMin = min(xCenter + (w / 2), xCenter - (w / 2)),
                                    xMax = max(xCenter + (w / 2), xCenter - (w / 2)),
                                    yMin = min(yCenter + (h / 2), yCenter - (h / 2)),
                                    yMax = max(yCenter + (h / 2), yCenter - (h / 2))
                                )
                                foundObjects.add(element)
                            }
                        }
                    }
                }
            }
        }

        return foundObjects.groupBy { it.classLabel }
            .mapValues { it.value.maxByOrNull { it.probability } }.values.filterNotNull().take(topK)
    }
}

fun main() = Yolov4.predict()