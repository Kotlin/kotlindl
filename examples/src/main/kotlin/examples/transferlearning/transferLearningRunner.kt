/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3L
private const val TRAIN_TEST_SPLIT_RATIO = 0.7

fun runImageRecognitionTransferLearning(
    modelType: TFModels.CV<out GraphTrainableModel>,
    epochs: Int = 2
) {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val dogsCatsImages = dogsCatsSmallDatasetPath()

    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = File(dogsCatsImages)
            imageShape = ImageShape(channels = NUM_CHANNELS)
            labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
        }
        transformImage {
            resize {
                outputHeight = modelType.inputShape?.get(0) ?: 224
                outputWidth = modelType.inputShape?.get(0) ?: 224
                interpolation = InterpolationType.BILINEAR
            }
            convert { colorMode = ColorMode.BGR }
        }
        transformTensor {
            sharpen {
                modelTypePreprocessing = modelType
            }
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    val hdfFile = modelHub.loadWeights(modelType)
    val layers = mutableListOf<Layer>()

    for (layer in model.layers) {
        layers.add(layer)
    }

    val newGlobalAvgPool2DLayer = GlobalAvgPool2D(
        name = "top_avg_pool",

        )
    newGlobalAvgPool2DLayer.inboundLayers.add(layers.last())
    layers.add(
        newGlobalAvgPool2DLayer
    )

    val newDenseLayer = Dense(
        name = "top_dense",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 200,
        activation = Activations.Relu
    )
    newDenseLayer.inboundLayers.add(layers.last())
    layers.add(
        newDenseLayer
    )

    val newDenseLayer2 = Dense(
        name = "pred",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = NUM_CLASSES,
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
        model2.logSummary()

        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = epochs
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}

fun runImageRecognitionTransferLearningOnTopModel(
    modelType: TFModels.CV<out GraphTrainableModel>,
    epochs: Int = 2
) {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val dogsCatsImages = dogsCatsSmallDatasetPath()

    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = File(dogsCatsImages)
            imageShape = ImageShape(channels = NUM_CHANNELS)
            labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
        }
        transformImage {
            resize {
                outputHeight = modelType.inputShape?.get(0) ?: 224
                outputWidth = modelType.inputShape?.get(0) ?: 224
                interpolation = InterpolationType.BILINEAR
            }
            convert { colorMode = ColorMode.BGR }
        }
        transformTensor {
            sharpen {
                modelTypePreprocessing = modelType
            }
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    val hdfFile = modelHub.loadWeights(modelType)
    val layers = mutableListOf<Layer>()

    for (layer in model.layers) {
        layer.isTrainable = false
        layers.add(layer)
    }

    val lastLayer = layers.last()
    for (outboundLayer in lastLayer.inboundLayers)
        outboundLayer.outboundLayers.remove(lastLayer)

    layers.removeLast()

    var x = Dense(
        name = "top_dense",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 200,
        activation = Activations.Relu
    )(layers.last())

    x = Dense(
        name = "pred",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = NUM_CLASSES,
        activation = Activations.Linear
    )(x)

    val model2 = Functional.fromOutput(x)

    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        model2.logSummary()

        val weightPaths = listOf(
            LayerConvOrDensePaths(
                "conv1_conv",
                "/conv1/conv/conv1/conv/kernel:0",
                "/conv1/conv/conv1/conv/bias:0"
            ),
            LayerBatchNormPaths(
                "conv1_bn",
                "/conv1/bn/conv1/bn/gamma:0",
                "/conv1/bn/conv1/bn/beta:0",
                "/conv1/bn/conv1/bn/moving_mean:0",
                "/conv1/bn/conv1/bn/moving_variance:0"
            )
        )

        it.loadWeightsByPaths(hdfFile, weightPaths, missedWeights = MissedWeightsStrategy.LOAD_CUSTOM_PATH, forFrozenLayersOnly = true)

        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = epochs
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}
