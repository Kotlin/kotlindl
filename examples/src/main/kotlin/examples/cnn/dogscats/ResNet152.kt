/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.dogscats

import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.model.resnet152Light
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val EPOCHS = 20
private const val TRAINING_BATCH_SIZE = 16
private const val TEST_BATCH_SIZE = 32
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 200L
private const val TRAIN_TEST_SPLIT_RATIO = 0.8

fun resnet152onDogsVsCatsDataset() {
    val dogsVsCatsDatasetPath = dogsCatsDatasetPath()

    val preprocessing: Preprocessing = preprocess {
        transformImage {
            load {
                pathToData = File(dogsVsCatsDatasetPath)
                imageShape = ImageShape(channels = NUM_CHANNELS)
                colorMode = ColorOrder.BGR
                labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
            }
            resize {
                outputHeight = IMAGE_SIZE.toInt()
                outputWidth = IMAGE_SIZE.toInt()
                interpolation = InterpolationType.BILINEAR
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    resnet152Light(imageSize = IMAGE_SIZE, numberOfClasses = NUM_CLASSES).use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val start = System.currentTimeMillis()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)
        println("Training time: ${(System.currentTimeMillis() - start) / 1000f}")

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = resnet152onDogsVsCatsDataset()
