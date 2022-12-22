/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Path

private const val EPS = 0.3
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 2000
private const val TEST_BATCH_SIZE = 1000
private const val AMOUNT_OF_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private val kernelInitializer = HeNormal(12L)
private val biasInitializer = HeUniform(12L)

class TensorFlowInferenceModelTest {
    private val lenet5Layers = listOf(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS,
            name = "input_0"
        ),
        Conv2D(
            filters = 32,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = kernelInitializer,
            biasInitializer = biasInitializer,
            padding = ConvPadding.SAME,
            name = "conv2d_1"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_2"
        ),
        Conv2D(
            filters = 64,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = kernelInitializer,
            biasInitializer = biasInitializer,
            padding = ConvPadding.SAME,
            name = "conv2d_3"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_4"
        ),
        Flatten(name = "flatten_5"), // 3136
        Dense(
            outputSize = 120,
            activation = Activations.Relu,
            kernelInitializer = kernelInitializer,
            biasInitializer = biasInitializer,
            name = "dense_6"
        ),
        Dense(
            outputSize = 84,
            activation = Activations.Relu,
            kernelInitializer = kernelInitializer,
            biasInitializer = biasInitializer,
            name = "dense_7"
        ),
        Dense(
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = kernelInitializer,
            biasInitializer = biasInitializer,
            name = "dense_8"
        )
    )

    @Test
    fun basicInference(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = mnist()

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val inferenceModel = TensorFlowInferenceModel.load(tempDir.toFile(), loadOptimizerState = false)
        inferenceModel.use {
            it.reshape(28, 28, 1)
            var accuracy = 0.0
            val amountOfTestSet = 10000
            for (imageId in 0..amountOfTestSet) {
                val prediction = it.predict(train.getX(imageId))
                val softPrediction = it.predictSoftly(train.getX(imageId))
                assertEquals(10, softPrediction.size)

                if (prediction == train.getY(imageId).toInt())
                    accuracy += (1.0 / amountOfTestSet)
            }

            assertTrue(accuracy > 0.5)
        }
    }

    @Test
    fun basicInferenceAndCopy(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = mnist()

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.4)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val inferenceModel = TensorFlowInferenceModel.load(tempDir.toFile(), loadOptimizerState = false)

        var copiedInferenceModel: TensorFlowInferenceModel

        val firstAccuracy: Double
        val secondAccuracy: Double

        inferenceModel.use {
            it.reshape(28, 28, 1)
            var accuracy = 0.0
            val amountOfTestSet = 10000
            for (imageId in 0..amountOfTestSet) {
                val prediction = it.predict(train.getX(imageId))
                val softPrediction = it.predictSoftly(train.getX(imageId))
                assertEquals(10, softPrediction.size)

                if (prediction == train.getY(imageId).toInt())
                    accuracy += (1.0 / amountOfTestSet)
            }
            copiedInferenceModel = inferenceModel.copy("CopiedLenet")
            assertTrue(accuracy > 0.5)
            firstAccuracy = accuracy
        }

        copiedInferenceModel.use {
            var accuracy = 0.0
            val amountOfTestSet = 10000
            for (imageId in 0..amountOfTestSet) {
                val prediction = it.predict(train.getX(imageId))

                if (prediction == train.getY(imageId).toInt())
                    accuracy += (1.0 / amountOfTestSet)
            }
            assertTrue(accuracy > 0.5)
            secondAccuracy = accuracy
        }

        assertEquals(firstAccuracy, secondAccuracy, EPS)
    }

    @Test
    fun emptyInferenceModel() {
        val (train, _) = mnist()

        val inferenceModel = TensorFlowInferenceModel()
        inferenceModel.use {
            it.reshape(28, 28, 1)

            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(train.getX(0))
                }
            assertEquals(
                "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method.",
                exception.message
            )
        }
    }

    @Test
    fun missedReshapeFunction() {
        val (train, _) = mnist()

        val inferenceModel = TensorFlowInferenceModel()
        inferenceModel.use {
            val exception =
                Assertions.assertThrows(IllegalArgumentException::class.java) {
                    it.predict(train.getX(0))
                }
            assertEquals(
                "Model input shape is not defined. Call reshape() to set input shape.",
                exception.message
            )

            val exception2 =
                Assertions.assertThrows(IllegalArgumentException::class.java) {
                    it.predictSoftly(train.getX(0))
                }
            assertEquals(
                "Model input shape is not defined. Call reshape() to set input shape.",
                exception2.message
            )
        }
    }

    @Test
    fun createInferenceModelOnJSONConfig(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = mnist()

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val exception =
            Assertions.assertThrows(FileNotFoundException::class.java) {
                TensorFlowInferenceModel.load(tempDir.toFile())
            }
        assertEquals(
            "File 'graph.pb' is not found. This file must be in the model directory. It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.TF_GRAPH.",
            exception.message
        )
    }

    @Test
            /** Manually removed file 'variableNames.txt'. */
    fun createInferenceModelOnCorruptedVariableData(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = mnist()

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        File(tempDir.toFile().absolutePath + "/variableNames.txt").delete()

        val exception =
            Assertions.assertThrows(FileNotFoundException::class.java) {
                TensorFlowInferenceModel.load(tempDir.toFile())
            }
        assertEquals(
            "File 'variableNames.txt' is not found. This file must be in the model directory. It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES.",
            exception.message
        )
    }
}
