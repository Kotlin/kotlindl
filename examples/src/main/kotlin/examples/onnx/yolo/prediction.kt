/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelZoo
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.mnist
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/onnx/yolov4.onnx"

fun main() {
    val modelZoo = ModelZoo(commonModelDirectory = File("cache/pretrainedModels"), modelType = ModelType.MobileNet)
    val model = modelZoo.loadModel() as Functional

    val imageNetClassLabels = modelZoo.loadClassLabels()

    OnnxInferenceModel.load(PATH_TO_MODEL).use {
        println(it)

        it.reshape(416, 416, 3)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocess {
                transformImage {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                    resize {
                        outputHeight = 416
                        outputWidth = 416
                    }
                }
            }

            // TODO: currently, the whole model is loaded but not used for prediction, the preprocessing is used only
            // Correct preprocessing https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
            val inputData = modelZoo.preprocessInput(preprocessing().first, model.inputDimensions)

            val res = it.rawPredict(inputData)
            println(res)
            // TODO: interpret the results of YOLOv4 https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
            /*val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)


            // TODO:
            // 1) interpret YOLOv4 numbers
            // 2) load YOLOv4 to ModelZoo
            // 3) load coco.names to resources https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/classes/coco.names
            // 4) add YOLOv4 preprocessing for photos https://www.reddit.com/r/computervision/comments/mc6m87/how_do_you_add_a_class_to_coco_classes_in_yolo/
            // 5) add YOLOv4 post-processing for photos https://www.reddit.com/r/computervision/comments/mc6m87/how_do_you_add_a_class_to_coco_classes_in_yolo/
            // 6) draw image on batik or SWING with rectangles by coordinates https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
            // 7) write a tutorial for YOLOv4 + add info about transfer learning https://www.reddit.com/r/computervision/comments/mc6m87/how_do_you_add_a_class_to_coco_classes_in_yolo/
            println(top5.toString())*/
        }
    }
}
