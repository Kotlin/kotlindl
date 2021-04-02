/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.resnet.resnet50

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.io.File
import java.io.FileReader
import java.util.*

fun main() {
    val jsonConfigFile = getResNet50JSONConfigFile()
    val model = Functional.loadModelConfiguration(jsonConfigFile)

    val data = prepareCustomDatasetFromPaths(
        "C:\\Users\\zaleslaw\\Desktop\\diplodok_rex\\diplo_224_224",
        "C:\\Users\\zaleslaw\\Desktop\\diplodok_rex\\rex_224_224"
    )

    val (train, test) = data.split(0.8)
    val hdfFile = getResNet50WeightsFile()
    val layers = mutableListOf<Layer>()

    for (layer in model.layers) {
        layer.isTrainable = false
        layers.add(layer)
    }

    layers.removeLast()

    val newDenseLayer = Dense(
        name = "top_dense",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 200,
        activation = Activations.Relu
    )
    newDenseLayer.inboundLayers.add(layers.last()) // bound to AveragePooling (TODO: better via API resnet(newDenseLayer))
    layers.add(
        newDenseLayer
    )

    val newDenseLayer2 = Dense(
        name = "pred",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 2,
        activation = Activations.Linear
    )
    newDenseLayer2.inboundLayers.add(layers.last())

    layers.add(
        newDenseLayer2
    )

    val model2 = Functional.of(layers)

    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.loadWeightsForFrozenLayers(hdfFile)
        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = 4).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = 8,
            epochs = 10
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 4).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }

}


fun prepareCustomDatasetFromPaths(vararg paths: String): OnHeapDataset {
    /*val listOfImages = mutableListOf<FloatArray>()
    val listOfLabels = mutableListOf<FloatArray>()
    val numberOfClasses = paths.size
    var counter = 0

    for (path in paths) {
        File(path).walk().forEach {
            try {
                val rawImage = ImageConverter.toRawFloatArray(it)

                val tensorShape = longArrayOf(
                    224,
                    224,
                    3
                )

                val image = preprocessInput(rawImage, tensorShape, inputType = InputType.CAFFE)


                listOfImages.add(image)
                val label = FloatArray(numberOfClasses)
                label[counter] = 1F
                listOfLabels.add(label)
            } catch (e: Exception) {
                println("Skipping the following image $it")
            }
        }
        counter += 1
    }

    val sortedData = listOfImages.zip(listOfLabels)
    val shuffledData = sortedData.shuffled()
    val (x, y) = shuffledData.unzip()

    return OnHeapDataset.create({ x.toTypedArray() }, { y.toTypedArray() })*/
    TODO()
    // unimplemented
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
private fun getResNet50JSONConfigFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val resnet50JSONModelPath = properties["resnet50JSONModelPath"] as String

    return File(resnet50JSONModelPath)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
private fun getResNet50WeightsFile(): HdfFile {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val resnet50h5WeightsPath = properties["resnet50h5WeightsPath"] as String

    return HdfFile(File(resnet50h5WeightsPath))
}


