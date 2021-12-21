/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg16


import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
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

private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 200
private const val TRAIN_TEST_SPLIT_RATIO = 0.7
private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val EPOCHS = 2

/**
 * This examples demonstrates the transfer learning concept on VGG'16 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - All layers, are added to the new Neural Network, its weights are frozen.
 * - New Dense layers are added and initialized via defined initializers.
 * - Model is re-trained on [dogsCatsSmallDatasetPath] dataset.
 *
 * We use the [Preprocessing] DSL to describe the dataset generation pipeline.
 * We demonstrate the workflow on the subset of Kaggle Cats vs Dogs binary classification dataset.
 *
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
 *    Detailed description of VGG'16 model and an approach to build it in Keras.</a>
 */
fun vgg16noTopAdditionalTraining() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.VGG16(noTop = true, inputShape = intArrayOf(IMAGE_SIZE, IMAGE_SIZE, 3))
    val model = modelHub.loadModel(modelType)

    val dogsVsCatsDatasetPath = dogsCatsSmallDatasetPath()

    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = File(dogsVsCatsDatasetPath)
            imageShape = ImageShape(channels = NUM_CHANNELS)
            labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
        }
        transformImage {
            resize {
                outputHeight = IMAGE_SIZE
                outputWidth = IMAGE_SIZE
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

    val layers = mutableListOf<Layer>()

    layers.addAll(model.layers)

    layers.add(
        Flatten(
            name = "flatten_1",
        )
    )
    layers.add(
        Dense(
            name = "new_dense_1",
            kernelInitializer = HeNormal(),
            biasInitializer = HeNormal(),
            outputSize = 64,
            activation = Activations.Relu,
        )
    )
    layers.add(
        Dense(
            name = "new_dense_2",
            kernelInitializer = HeNormal(),
            biasInitializer = HeNormal(),
            outputSize = 2,
            activation = Activations.Linear
        )
    )

    val newModel = Sequential.of(layers)

    newModel.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        val hdfFile = modelHub.loadWeights(modelType)
        it.loadWeightsForFrozenLayers(hdfFile)

        it.printSummary()

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
fun main(): Unit = vgg16noTopAdditionalTraining()


