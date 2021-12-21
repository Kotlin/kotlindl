/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.MAE
import org.jetbrains.kotlinx.dl.api.core.metric.MSE
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2L1
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.core.util.OUTPUT_NAME
import org.jetbrains.kotlinx.dl.dataset.handler.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.mnist
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

private const val INCORRECT_AMOUNT_OF_CLASSES_1 = 11

internal class SequentialBasicTest : IntegrationTest() {
    private val testModel = Sequential.of(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        ),
        Conv2D(
            filters = 32,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeUniform(SEED),
            padding = ConvPadding.SAME,
            name = "conv2d_1"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_1"
        ),
        Conv2D(
            filters = 64,
            kernelSize = intArrayOf(5, 5),
            strides = intArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeUniform(SEED),
            padding = ConvPadding.SAME,
            name = "conv2d_2",
            useBias = false
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_2"
        ),
        Flatten(name = "flatten_1"), // 3136
        Dense(
            outputSize = 512,
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeUniform(SEED),
            name = "dense_1"
        ),
        Dense(
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeUniform(SEED),
            name = "dense_2",
            useBias = false
        )
    )

    @Test
    fun mnistDatasetCreation() {
        val (train, test) = mnist()

        assertEquals(train.xSize(), 60000)
        assertEquals(test.xSize(), 10000)
    }


    @Test
    fun initLeNetModel() {
        val (train, test) = mnist()

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())
            it.init()

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.1)
            }
        }
    }


    @Test
    fun trainingLeNetModel() {
        val (train, test) = mnist()

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }

            it.logSummary()

            // Prediction testing
            val label = it.predict(test.getX(0))
            assertEquals(test.getY(0), label.toFloat())

            val softPrediction = it.predictSoftly(test.getX(0))

            assertEquals(
                test.getY(0),
                softPrediction.indexOfFirst { value -> value == softPrediction.maxOrNull()!! }.toFloat()
            )

            // Test predict method with specified tensor name
            val label2 = it.predict(test.getX(0), predictionTensorName = OUTPUT_NAME)
            assertEquals(test.getY(0), label2.toFloat())

            // Test predictAndGetActivations method
            val (label3, activations) = it.predictAndGetActivations(test.getX(0))
            assertEquals(test.getY(0), label3.toFloat())
            assertEquals(3, activations.size)

            // TODO: flaky test due to TF non-determinism
            /*val conv2d1Activations = activations[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.43086472153663635, conv2d1Activations[0][0][0][0].toDouble(), EPS)
            val conv2d2Activations = activations[1] as Array<Array<Array<FloatArray>>>
            assertEquals(0.0, conv2d2Activations[0][0][0][0].toDouble(), EPS)
            val denseActivations = activations[2] as Array<FloatArray>
            assertEquals(2.8752777576446533, denseActivations[0][0].toDouble(), EPS)*/

            val predictions = it.predict(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), predictions.size)

            val softPredictions = it.predictSoftly(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), softPredictions.size)
            assertEquals(AMOUNT_OF_CLASSES, softPredictions[0].size)

            var manualAccuracy = 0
            predictions.forEachIndexed { index, lb -> if (lb == test.getY(index).toInt()) manualAccuracy++ }
            assertTrue(manualAccuracy > 0.7)
        }
    }

    @Test
    fun trainingLeNetModelWithThreeMetrics() {
        val (train, test) = mnist()

        testModel.use {
            it.compile(optimizer = Adam(), loss = SoftmaxCrossEntropyWithLogits(), metrics = listOf(Accuracy(), MSE(), MAE()))

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[1] > 0.05f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[2] > 0.05f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            val mse = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MSE]
            val mae = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MAE]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }
            if (mse != null) {
                assertTrue(mse < 0.02)
            }
            if (mae != null) {
                assertTrue(mae < 0.02)
            }

            it.logSummary()

            // Prediction testing
            val label = it.predict(test.getX(0))
            assertEquals(test.getY(0), label.toFloat())

            val softPrediction = it.predictSoftly(test.getX(0))

            assertEquals(
                test.getY(0),
                softPrediction.indexOfFirst { value -> value == softPrediction.maxOrNull()!! }.toFloat()
            )

            // Test predict method with specified tensor name
            val label2 = it.predict(test.getX(0), predictionTensorName = OUTPUT_NAME)
            assertEquals(test.getY(0), label2.toFloat())

            // Test predictAndGetActivations method
            val (label3, activations) = it.predictAndGetActivations(test.getX(0))
            assertEquals(test.getY(0), label3.toFloat())
            assertEquals(3, activations.size)

            // TODO: flaky test due to TF non-determinism
            /*val conv2d1Activations = activations[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.43086472153663635, conv2d1Activations[0][0][0][0].toDouble(), EPS)
            val conv2d2Activations = activations[1] as Array<Array<Array<FloatArray>>>
            assertEquals(0.0, conv2d2Activations[0][0][0][0].toDouble(), EPS)
            val denseActivations = activations[2] as Array<FloatArray>
            assertEquals(2.8752777576446533, denseActivations[0][0].toDouble(), EPS)*/

            val predictions = it.predict(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), predictions.size)

            val softPredictions = it.predictSoftly(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), softPredictions.size)
            assertEquals(AMOUNT_OF_CLASSES, softPredictions[0].size)

            var manualAccuracy = 0
            predictions.forEachIndexed { index, lb -> if (lb == test.getY(index).toInt()) manualAccuracy++ }
            assertTrue(manualAccuracy > 0.7)
        }
    }

    @Test
    fun trainAndCopyLeNetModel() {
        // TODO: add the same test for the Functional model, ResNet or ToyResNet
        val (train, test) = mnist()

        var copiedModel: Sequential
        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }

            it.logSummary()

            // Prediction testing
            val label = it.predict(test.getX(0))
            assertEquals(test.getY(0), label.toFloat())

            val softPrediction = it.predictSoftly(test.getX(0))

            assertEquals(
                test.getY(0),
                softPrediction.indexOfFirst { value -> value == softPrediction.maxOrNull()!! }.toFloat()
            )

            // Test predict method with specified tensor name
            val label2 = it.predict(test.getX(0), predictionTensorName = OUTPUT_NAME)
            assertEquals(test.getY(0), label2.toFloat())

            // Test predictAndGetActivations method
            val (label3, activations) = it.predictAndGetActivations(test.getX(0))
            assertEquals(test.getY(0), label3.toFloat())
            assertEquals(3, activations.size)

            // TODO: flaky asserts due-to non-determinism
            /*val conv2d1Activations = activations[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.43086472153663635, conv2d1Activations[0][0][0][0].toDouble(), EPS)
            val conv2d2Activations = activations[1] as Array<Array<Array<FloatArray>>>
            assertEquals(0.0, conv2d2Activations[0][0][0][0].toDouble(), EPS)
            val denseActivations = activations[2] as Array<FloatArray>
            assertEquals(2.8752777576446533, denseActivations[0][0].toDouble(), EPS)*/

            val predictions = it.predict(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), predictions.size)

            val softPredictions = it.predictSoftly(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), softPredictions.size)
            assertEquals(AMOUNT_OF_CLASSES, softPredictions[0].size)

            var manualAccuracy = 0
            predictions.forEachIndexed { index, lb -> if (lb == test.getY(index).toInt()) manualAccuracy++ }
            assertTrue(manualAccuracy > 0.7)

            copiedModel = it.copy(copyWeights = true)

            copiedModel.layers.forEach { layer ->
                run {
                    val weights = copiedModel.getLayer(layer.name).weights
                    weights.forEach { (varName, arr) ->
                        assertArrayEquals(arr, it.getLayer(layer.name).weights[varName]) //arr.contentDeepEquals()
                    }
                }
            }
        }

        copiedModel.use {
            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }
        }
    }

    @Test
    fun trainingWithValidation() {
        val (train, test) = mnist()
        val (newTrain, validation) = train.split(0.95)

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val trainingHistory =
                it.fit(
                    trainingDataset = newTrain,
                    validationDataset = validation,
                    epochs = EPOCHS,
                    trainBatchSize = TRAINING_BATCH_SIZE,
                    validationBatchSize = TEST_BATCH_SIZE
                )

            assertEquals(57, trainingHistory.batchHistory.size)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }
        }
    }

    @Test
    fun trainingFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        epochs = EPOCHS,
                        batchSize = TRAINING_BATCH_SIZE
                    )
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun evaluatingFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun predictionFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.predict(train.getX(0))
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun softPredictionFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.predictSoftly(train.getX(0))
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun predictAllFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalArgumentException::class.java) {
                    it.predict(test, 256)
                }
            assertEquals(
                "The amount of images must be a multiple of batch size.",
                exception.message
            )
        }

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.predict(test, 100)
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun predictAndGetActivationsFailedWithoutCompilation() {
        val (train, test) = mnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.predictAndGetActivations(test.getX(0))
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun allActivationsAreBelongToUs() {
        val testModel = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Selu,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                padding = ConvPadding.SAME,
                name = "conv2d_1"
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "maxPool_1"
            ),
            Conv2D(
                filters = 64,
                kernelSize = intArrayOf(3, 3),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Elu,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                padding = ConvPadding.SAME,
                name = "conv2d_2"
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "maxPool_2"
            ),
            Flatten(name = "flatten_1"), // 3136
            Dense(
                outputSize = 64,
                activation = Activations.Relu6,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_1"
            ),
            Dense(
                outputSize = 32,
                activation = Activations.Swish,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_2"
            ),
            Dense(
                outputSize = 32,
                activation = Activations.SoftPlus,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_3"
            ),
            Dense(
                outputSize = 32,
                activation = Activations.HardSigmoid,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_4"
            ),
            Dense(
                outputSize = 32,
                activation = Activations.LogSoftmax,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_5"
            ),
            Dense(
                outputSize = 32,
                activation = Activations.Exponential,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_6"
            ),
            Dense(
                outputSize = 10,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(),
                biasInitializer = HeNormal(),
                name = "dense_7"
            )
        )

        val (train, test) = mnist()

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            it.logSummary()

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 100.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.05)
            }
        }

    }

    @Test
    fun rarelyUsedBuildingBlocks() {
        val testModel = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelInitializer = HeNormal(13L),
                biasInitializer = HeNormal(13L),
                padding = ConvPadding.VALID,
                name = "conv2d_1"
            ),
            AvgPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "avgPool_1"
            ),
            Conv2D(
                filters = 64,
                kernelInitializer = GlorotNormal(13L),
                biasInitializer = GlorotUniform(13L),
                padding = ConvPadding.SAME,
                name = "conv2d_2"
            ),
            AvgPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "maxPool_2"
            ),
            Flatten(name = "flatten_1"), // 3136
            Dense(
                outputSize = 64,
                kernelInitializer = LeCunNormal(13L),
                biasInitializer = LeCunUniform(13L),
                name = "dense_1"
            ),
            Dropout(
                keepProbability = 0.3f,
                seed = 13L,
                name = "dropout_1"
            ),
            Dense(
                outputSize = 32,
                name = "dense_2"
            ),
            Dense(
                outputSize = 10,
                activation = Activations.Linear,
                name = "dense_3"
            )
        )

        val (train, test) = mnist()

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            it.logSummary()

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.05)
            }
        }
    }

    @Test
    fun lenetWithRegularization() {
        val testModel = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = intArrayOf(5, 5),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = HeNormal(SEED),
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1)
            ),
            Conv2D(
                filters = 64,
                kernelSize = intArrayOf(5, 5),
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = HeNormal(SEED),
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1)
            ),
            Flatten(), // 3136
            Dense(
                outputSize = 512,
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = HeNormal(SEED),
                kernelRegularizer = L2L1(),
                biasRegularizer = L2L1(),
            ),
            Dense(
                outputSize = NUMBER_OF_CLASSES,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = HeNormal(SEED),
                kernelRegularizer = L2L1(),
                biasRegularizer = L2L1(),
            )
        )

        val (train, test) = mnist()

        testModel.use {
            it.compile(
                optimizer = SGD(learningRate = 0.1f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.logSummary()

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = 1000)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValues[0] < 0.2f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.05)
            }
        }
    }
}
