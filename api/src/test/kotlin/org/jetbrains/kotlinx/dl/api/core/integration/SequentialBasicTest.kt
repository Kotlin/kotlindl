/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.util.OUTPUT_NAME
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
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
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Zeros(),
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
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Ones(),
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
            outputSize = 512,
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_1"
        ),
        Dense(
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeUniform(SEED),
            name = "dense_2"
        )
    )

    @Test
    fun mnistDatasetCreation() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        assertEquals(train.xSize(), 60000)
        assertEquals(test.xSize(), 10000)
    }


    @Test
    fun initLeNetModel() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = false)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValue < 0.2f)

            // Evaluation testing
            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }

            it.summary()

            // Prediction testing
            val label = it.predict(test.getX(0))
            assertEquals(test.getLabel(0), label)

            val softPrediction = it.predictSoftly(test.getX(0))

            assertEquals(
                test.getLabel(0),
                softPrediction.indexOfFirst { value -> value == softPrediction.maxOrNull()!! })

            // Test predict method with specified tensor name
            val label2 = it.predict(test.getX(0), predictionTensorName = OUTPUT_NAME)
            assertEquals(test.getLabel(0), label2)

            // Test predictAndGetActivations method
            val (label3, activations) = it.predictAndGetActivations(test.getX(0))
            assertEquals(test.getLabel(0), label3)
            assertEquals(3, activations.size)

            val conv2d1Activations = activations[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2d1Activations[0][0][0][0].toDouble(), 0.00690492382273078, EPS)
            val conv2d2Activations = activations[1] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2d2Activations[0][0][0][0].toDouble(), 0.9169542789459229, EPS)
            val denseActivations = activations[2] as Array<FloatArray>
            assertEquals(denseActivations[0][0].toDouble(), 0.0, EPS)

            val predictions = it.predict(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), predictions.size)

            val softPredictions = it.predictSoftly(test, TEST_BATCH_SIZE)
            assertEquals(test.xSize(), softPredictions.size)
            assertEquals(AMOUNT_OF_CLASSES, softPredictions[0].size)

            var manualAccuracy = 0
            predictions.forEachIndexed { index, lb -> if (lb == test.getLabel(index)) manualAccuracy++ }
            assertTrue(manualAccuracy > 0.7)
        }
    }

    @Test
    fun trainingWithValidation() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )
        val (newTrain, validation) = train.split(0.95)

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val trainingHistory =
                it.fit(
                    trainingDataset = newTrain,
                    validationDataset = validation,
                    epochs = EPOCHS,
                    trainBatchSize = TRAINING_BATCH_SIZE,
                    validationBatchSize = TEST_BATCH_SIZE,
                    verbose = true
                )

            assertEquals(57, trainingHistory.batchHistory.size)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 2.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValue < 0.2f)

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }
        }
    }

    @Test
    fun trainingFailedWithoutCompilation() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        epochs = EPOCHS,
                        batchSize = TRAINING_BATCH_SIZE,
                        verbose = false
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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalArgumentException::class.java) {
                    it.predict(test, 256)
                }
            assertEquals(
                "The amount of images must be a multiple of batch size.",
                exception.message
            )
        }

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
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
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
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
                kernelSize = longArrayOf(3, 3),
                strides = longArrayOf(1, 1, 1, 1),
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
                kernelSize = longArrayOf(3, 3),
                strides = longArrayOf(1, 1, 1, 1),
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

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            it.summary()

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(1, trainingHistory.batchHistory[0].epochIndex)
            assertEquals(0, trainingHistory.batchHistory[0].batchIndex)
            assertTrue(trainingHistory.batchHistory[0].lossValue > 100.0f)
            assertTrue(trainingHistory.batchHistory[0].metricValue < 0.2f)

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

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            it.summary()

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

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
    fun incorrectAmountOfClassesInTheLastDenseLayer() {
        val testModelWithSmallAmountOfClasses = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = longArrayOf(5, 5),
                strides = longArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Zeros(),
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
                kernelSize = longArrayOf(5, 5),
                strides = longArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Zeros(),
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
                outputSize = 512,
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f),
                name = "dense_1"
            ),
            Dense(
                outputSize = INCORRECT_AMOUNT_OF_CLASSES_1,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f),
                name = "dense_2"
            )
        )


        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModelWithSmallAmountOfClasses.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        epochs = EPOCHS,
                        batchSize = TRAINING_BATCH_SIZE,
                        verbose = false
                    )
                }
            assertEquals(
                "The calculated [from the Sequential model] label batch shape [1000, 11] doesn't match actual data buffer size 10000. \n" +
                        "Please, check the input label data or correct amount of classes [amount of neurons] in last Dense layer, if you have a classification problem.\n" +
                        "Highly likely, you have different amount of classes presented in data and described in model as desired output.",
                exception.message
            )
        }
    }
}
