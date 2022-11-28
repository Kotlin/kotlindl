/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.resnet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTop5Labels
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/resnet50_1"
private const val PATH_TO_MODEL_2 = "savedmodels/resnet50_2"

/**
 * This examples demonstrates the inference concept on ResNet'50 model and model, model weight export and import back:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in ResNet'50 during training on ImageNet dataset) is applied to each image before prediction.
 * - Model is exported in  both: Keras-style JSON format and graph .pb format ; weights are exported in custom (TXT) format.
 * - It saves all the data to the project root directory.
 * - The first [TensorFlowInferenceModel] is created via graph and weights loading.
 * - Model again predicts on a few images located in resources.
 * - The second [Functional] model is created via JSON configuration and weights loading.
 * - Model again predicts on a few images located in resources.
 */
fun main() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.ResNet50()
    val model = modelHub.loadModel(modelType)

    val fileDataLoader = pipeline<BufferedImage>()
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .call(modelType.preprocessor)
        .fileLoader()

    val imageNetClassLabels = modelHub.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = modelHub.loadWeights(modelType)

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }

        it.save(
            File(PATH_TO_MODEL),
            savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )

        it.save(
            File(PATH_TO_MODEL_2),
            savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )
    }

    val inferenceModel = TensorFlowInferenceModel.load(File(PATH_TO_MODEL_2))

    inferenceModel.use {
        for (i in 1..8) {
            it.reshape(224, 224, 3)

            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }

    val model2 = Functional.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    model2.use {
        it.compile(
            optimizer = RMSProp(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        it.logSummary()

        it.loadWeights(File(PATH_TO_MODEL))

        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

