/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.Identity
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTop5Labels
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.Resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File
import java.net.URISyntaxException
import java.net.URL

/** Converts resource string path to the file. */
@Throws(URISyntaxException::class)
fun getFileFromResource(fileName: String): File {
    val classLoader: ClassLoader = object {}.javaClass.classLoader
    val resource: URL? = classLoader.getResource(fileName)
    return if (resource == null) {
        throw IllegalArgumentException("file not found! $fileName")
    } else {
        File(resource.toURI())
    }
}

fun runImageRecognitionPrediction(
    modelType: TFModels.CV<out GraphTrainableModel>,
    resizeTo: Pair<Int, Int> = Pair(224, 224)
) {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = modelHub.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.printSummary()
        it.logSummary()

        val hdfFile = modelHub.loadWeights(modelType)

        it.loadWeights(hdfFile)

        val fileDataLoader = fileDataLoader(modelType, resizeTo)
        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

internal fun fileDataLoader(
    modelType: ModelType<*, *>,
    resizeTo: Pair<Int, Int>
): DataLoader<File> {
    val resize = if (resizeTo.first == 224 && resizeTo.second == 224) {
        Identity()
    } else {
        Resize(
            outputWidth = resizeTo.first,
            outputHeight = resizeTo.second
        )
    }
    return pipeline<BufferedImage>()
        .call(resize)
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .call(modelType.preprocessor)
        .fileLoader()
}
