/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.resnet

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels.CV.Companion.createPreprocessing
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.generator.FromFolders
import java.io.File

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3
private const val IMAGE_SIZE = 300
private const val TRAIN_TEST_SPLIT_RATIO = 0.7

/**
 * This examples demonstrates the transfer learning concept on ResNet'50 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - All layers, excluding the last [Dense], are added to the new Neural Network, its weights are frozen.
 * - New Dense layers are added and initialized via defined initializers.
 * - Model is re-trained on [dogsCatsSmallDatasetPath] dataset.
 *
 * We use the preprocessing DSL to describe the dataset generation pipeline.
 * We demonstrate the workflow on the subset of Kaggle Cats vs Dogs binary classification dataset.
 */
fun resnet50additionalTrainingNoTopWithHelper() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.ResNet50(noTop = true, inputShape = intArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    val noTopModel = modelHub.loadModel(modelType)

    val hdfFile = modelHub.loadWeights(modelType)

    val topModel = Sequential.of(
        GlobalAvgPool2D(
            name = "top_avg_pool",
        ),
        Dense(
            name = "top_dense",
            kernelInitializer = GlorotUniform(),
            biasInitializer = GlorotUniform(),
            outputSize = 200,
            activation = Activations.Relu
        ),
        Dense(
            name = "pred",
            kernelInitializer = GlorotUniform(),
            biasInitializer = GlorotUniform(),
            outputSize = NUM_CLASSES,
            activation = Activations.Linear
        ),
        noInput = true
    )

    val model = Functional.of(pretrainedModel = noTopModel, topModel = topModel)

    val dataset = OnFlyImageDataset.create(
        File(dogsCatsSmallDatasetPath()),
        FromFolders(mapping = mapOf("cat" to 0, "dog" to 1)),
        modelType.createPreprocessing(model)
    ).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.loadWeightsForFrozenLayers(hdfFile)
        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = EPOCHS
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}

/** */
fun main(): Unit = resnet50additionalTrainingNoTopWithHelper()
