/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/onnx/ssd.onnx"

fun main() {
    val modelHub = ONNXModelHub(commonModelDirectory = File("cache/pretrainedModels"), modelType = ONNXModels.ObjectDetection.SSD)
    val model = modelHub.loadModel() as OnnxInferenceModel

    model.use {
        println(it)

        for (i in 0..8) {
            val preprocessing: Preprocessing = preprocess {
                transformImage {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                    resize {
                        outputHeight = 1200
                        outputWidth = 1200
                    }
                }
                transformTensor {
                    sharpen {
                        modelType = TFModels.CV.DenseNet201
                    }
                    transpose {
                        axes = intArrayOf(2, 0, 1)
                    }
                }
            }

            // TODO: currently, the whole model is loaded but not used for prediction, the preprocessing is used only
            // Correct preprocessing https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
            val inputData = modelHub.preprocessInput(preprocessing().first, model.inputDimensions)

            val yhat = it.predictRaw(inputData)
            println(yhat)
            /*val smallGrid = (yhat as List<Array<*>>)[2][0] as Array<Array<Array<FloatArray>>>
            for (i in 0..12){
                for (j in 0..12) {
                    for (k in 0..2) {
                        val row = smallGrid[i][j][k]
                        if (row[4] > 0.002f)
                            println(cocoCategories[row.copyOfRange(5, row.size).toList().argmax()])
                        // TODO: it returns the first index, if probabilites are equal it will return the one value for vase or book every time
                    }
                }

            }*/
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

