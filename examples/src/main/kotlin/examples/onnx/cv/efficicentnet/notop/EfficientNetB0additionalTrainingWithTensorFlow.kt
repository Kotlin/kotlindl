/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv.efficicentnet.notop

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsDatasetPath
import org.jetbrains.kotlinx.dl.dataset.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 64
private const val TEST_BATCH_SIZE = 32
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 224L
private const val TRAIN_TEST_SPLIT_RATIO = 0.8

/**
 * This is a simple model based on Dense layers only.
 */
private val topModel = Sequential.of(
    Input(7, 7, 1280),
    GlobalAvgPool2D(),
    Dense(NUM_CLASSES, Activations.Linear, kernelInitializer = HeNormal(12L), biasInitializer = Zeros())
)

/**
 * This examples demonstrates the transfer learning concept on EfficientNetB0 model:
 * - Model configuration, model weights and labels are obtained from [ONNXModelHub].
 * - All layers, excluding the last [Dense], are added to the new Neural Network, its weights are frozen.
 * - ONNX frozen model is used as a preprocessing stage via `onnx` stage of the Image Preprocessing DSL.
 * - New Dense layers are added and initialized via defined initializers.
 * - Model is re-trained on [dogsCatsDatasetPath] dataset.
 *
 *
 * We use the [Preprocessing] DSL to describe the dataset generation pipeline.
 * We demonstrate the workflow on the subset of Kaggle Cats vs Dogs binary classification dataset.
 */
fun efficientNetB0AdditionalTraining() {
    val modelHub = ONNXModelHub(
        cacheDirectory = File("cache/pretrainedModels")
    )
    val model = modelHub.loadModel(ONNXModels.CV.EfficientNetB0(noTop = true))

    model.use {
        println(it)
        val dogsVsCatsDatasetPath = dogsCatsSmallDatasetPath()

        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = File(dogsVsCatsDatasetPath)
                imageShape = ImageShape(channels = NUM_CHANNELS)
                labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
            }
            transformImage {
                resize {
                    outputHeight = IMAGE_SIZE.toInt()
                    outputWidth = IMAGE_SIZE.toInt()
                    interpolation = InterpolationType.BILINEAR
                }
                convert { colorMode = ColorMode.BGR }
            }
            transformTensor {
                onnx {
                    onnxModel = model
                }
            }
        }

        val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
        val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

        topModel.use {
            topModel.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            topModel.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            val accuracy = topModel.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")
        }
    }
}

/** */
fun main(): Unit = efficientNetB0AdditionalTraining()


