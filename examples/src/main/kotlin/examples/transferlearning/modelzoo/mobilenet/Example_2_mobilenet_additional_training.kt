/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning


import examples.transferlearning.modelzoo.resnet.resnet50.prepareCustomDatasetFromPaths
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelLoader
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import java.io.File

/**
 * This examples demonstrates the inference concept on MobileNet model:
 *
 * Weights are loaded from .h5 file, configuration is loaded from .json file.
 *
 * Model predicts on a few images located in resources.
 */
fun main() {
    val modelLoader =
        ModelLoader(commonModelDirectory = File("savedmodels/keras_models"), modelType = ModelType.MobileNet)
    val model = modelLoader.loadModel() as Functional

    val data = prepareCustomDatasetFromPaths(
        "C:\\Users\\zaleslaw\\Desktop\\diplodok_rex\\diplo_224_224",
        "C:\\Users\\zaleslaw\\Desktop\\diplodok_rex\\rex_224_224"
    )

    val (train, test) = data.split(0.8)

    val hdfFile = modelLoader.loadWeights()

    model.use {
        it.layers.last().isTrainable = true

        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()
    }

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

        it.summary()

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
