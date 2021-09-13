/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.yolo

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/onnx/yolov4.onnx"

// TODO: this example doesn't work
fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.MobileNet
    val model = modelHub.loadModel(modelType)

    OnnxInferenceModel.load(PATH_TO_MODEL).use {
        println(it)

        for (i in 0..8) {
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
                transformTensor {
                    rescale {
                        scalingCoefficient = 255.0f
                    }
                }
            }

            // TODO: currently, the whole model is loaded but not used for prediction, the preprocessing is used only
            // Correct preprocessing https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
            val inputData = modelType.preprocessInput(preprocessing().first, model.inputDimensions)

            val yhat = it.predictRaw(inputData)
            val smallGrid = yhat[2][0] as Array<Array<Array<FloatArray>>>
            for (i in 0..12) {
                for (j in 0..12) {
                    for (k in 0..2) {
                        val row = smallGrid[i][j][k]
                        if (row[4] > 0.002f)
                            println(cocoCategories[row.copyOfRange(5, row.size).toList().argmax()])
                        // TODO: it returns the first index, if probabilites are equal it will return the one value for vase or book every time
                    }
                }

            }
            // TODO: interpret the results of YOLOv4 https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
            /*val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)


            // TODO:
            // 1) interpret YOLOv4 numbers
            // 2) load YOLOv4 to ModelZoo
            // 4) add YOLOv4 preprocessing for photos https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4#preprocessing-steps
            // 5) add YOLOv4 post-processing for photos https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4#preprocessing-steps
            // 6) draw image on batik or SWING with rectangles by coordinates https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
            // 7) write a tutorial for YOLOv4 + add info about transfer learning https://www.reddit.com/r/computervision/comments/mc6m87/how_do_you_add_a_class_to_coco_classes_in_yolo/
            // 8) interpritation of yhat is here https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
            println(top5.toString())*/
        }
    }
}


fun <T : Comparable<T>> Iterable<T>.argmax(): Int? {
    return withIndex().maxByOrNull { it.value }?.index
}
