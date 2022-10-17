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
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.isTrainable
import org.jetbrains.kotlinx.dl.api.core.layer.paramCount
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.*
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

private const val EPS = 0.1
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 2000
private const val TEST_BATCH_SIZE = 1000
private const val AMOUNT_OF_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private val kernelInitializer = HeNormal(12L)
private val biasInitializer = HeUniform(12L)

class SequentialInferenceTest {
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
                println("Accuracy is $accuracy")
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val model = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        model.use {
            assertEquals(model.layers.size, 9)
            assertTrue(model.getLayer("input_0") is Input)
            assertTrue(model.getLayer("conv2d_1") is Conv2D)
            assertTrue(model.getLayer("conv2d_3") is Conv2D)
            assertTrue(model.getLayer("conv2d_1").isTrainable)
            assertTrue(model.getLayer("conv2d_1").hasActivation)
            Assertions.assertFalse(model.getLayer("flatten_5").isTrainable)
            Assertions.assertFalse(model.getLayer("flatten_5").hasActivation)
            assertTrue(model.getLayer("maxPool_2") is MaxPool2D)
            assertTrue(model.getLayer("maxPool_4") is MaxPool2D)
            assertTrue(model.getLayer("dense_6") is Dense)
            assertTrue(model.getLayer("dense_7") is Dense)
            assertTrue(model.getLayer("dense_8") is Dense)
            Assertions.assertArrayEquals(model.inputLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

            it.compile(
                optimizer = RMSProp(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            assertEquals(it.layers[0].paramCount, 0)
            assertEquals(it.layers[1].paramCount, 832)
            assertEquals(it.layers[2].paramCount, 0)
            assertEquals(it.layers[3].paramCount, 51264)
            assertEquals(it.layers[4].paramCount, 0)
            assertEquals(it.layers[5].paramCount, 0)
            assertEquals(it.layers[6].paramCount, 376440)
            assertEquals(it.layers[7].paramCount, 10164)
            assertEquals(it.layers[8].paramCount, 850)

            Assertions.assertArrayEquals(it.layers[0].outputShape.dims(), longArrayOf(-1, 28, 28, 1))
            Assertions.assertArrayEquals(it.layers[1].outputShape.dims(), longArrayOf(-1, 28, 28, 32))
            Assertions.assertArrayEquals(it.layers[2].outputShape.dims(), longArrayOf(-1, 14, 14, 32))
            Assertions.assertArrayEquals(it.layers[3].outputShape.dims(), longArrayOf(-1, 14, 14, 64))
            Assertions.assertArrayEquals(it.layers[4].outputShape.dims(), longArrayOf(-1, 7, 7, 64))
            Assertions.assertArrayEquals(it.layers[5].outputShape.dims(), longArrayOf(-1, 3136))
            Assertions.assertArrayEquals(it.layers[6].outputShape.dims(), longArrayOf(-1, 120))
            Assertions.assertArrayEquals(it.layers[7].outputShape.dims(), longArrayOf(-1, 84))
            Assertions.assertArrayEquals(it.layers[8].outputShape.dims(), longArrayOf(-1, 10))

            it.loadWeights(tempDir.toFile())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            if (accuracyBefore != null) {
                println("Accuracy before is $accuracyBefore")
                assertTrue(accuracyBefore > 0.5)
            }
        }
    }

    @Test
    fun failInferenceModelIfSomethingMissed(@TempDir tempDir: Path?) {
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
                println("Accuracy is $accuracy")
                assertTrue(accuracy > 0.2)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val model1 = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        // Test 1: Model is not initialized and is not compiled.
        model1.use {
            val evaluateException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.evaluate(dataset = test, batchSize = 100)
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                evaluateException.message
            )

            val fitException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        validationRate = 0.1,
                        epochs = EPOCHS,
                        trainBatchSize = TRAINING_BATCH_SIZE,
                        validationBatchSize = TEST_BATCH_SIZE
                    )
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                fitException.message
            )

            val predictException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(test.getX(0))
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                predictException.message
            )

            val predictAllException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(test, TEST_BATCH_SIZE)
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                predictAllException.message
            )
        }

        // Test 2: Model is compiled and is not initialized.
        val model2 = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        model2.use {
            it.compile(
                optimizer = RMSProp(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            val evaluateException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.evaluate(dataset = test, batchSize = 100)
                }
            assertEquals(
                "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method.",
                evaluateException.message
            )

            val predictException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(test.getX(0))
                }
            assertEquals(
                "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method.",
                predictException.message
            )

            val predictAllException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(test, TEST_BATCH_SIZE)
                }
            assertEquals(
                "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method.",
                predictAllException.message
            )
        }

        // Test 3: Model is initialized and is not compiled.
        val model3 = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        model3.use {
            val weightLoadingException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.loadWeights(tempDir.toFile())
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                weightLoadingException.message
            )

            val initMethodException =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.init()
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                initMethodException.message
            )
        }
    }

    @Test
    fun layerFreezingAndAdditionalTraining(@TempDir tempDir: Path?) {
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
                println("Accuracy is $accuracy")
                assertTrue(accuracy > 0.3)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }

        val model = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        model.use {
            it.layers.filterIsInstance<Conv2D>().forEach(Layer::freeze)

            it.compile(
                optimizer = RMSProp(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(tempDir.toFile())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            if (accuracyBefore != null) {
                println("Accuracy before is $accuracyBefore")
                assertTrue(accuracyBefore > 0.3)
            }

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            if (accuracyAfterTraining != null) {
                println("Accuracy after is $accuracyAfterTraining")
                assertTrue(accuracyAfterTraining > accuracyBefore!!)
            }
        }
    }

    @Test
    fun exportImportWithAdamOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Adam(), tempDir)
        assertTrue(testMetrics.getValue("trainAccuracy") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.8)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.8)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("afterAccuracy1"))
    }

    @Test
    fun exportImportWithAdaDeltaOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(AdaDelta(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.5)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.5)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.7)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.5)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.7)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("afterAccuracy1"))
    }

    @Test
    fun exportImportWithAdaGradOptimizerInternalState(@TempDir tempDir: Path?) {
        val testMetrics =
            trainingAndInferenceWithSpecificOptimizer(AdaGrad(clipGradient = ClipGradientByValue(0.01f)), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.3)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.3)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.7)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.3)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.7)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
    }

    @Test
    fun exportImportWithAdaGradDAOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(AdaGradDA(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.01)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.01)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.01)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.01)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.01)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
    }

    @Test
    fun exportImportWithAdamaxOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Adamax(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.8)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.8)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("afterAccuracy1"))
    }

    @Test
    fun exportImportWithFtrlOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Ftrl(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.7)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.7)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.7)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.8)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("afterAccuracy1"))
    }

    @Test
    fun exportImportWithMomentumOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Momentum(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.7)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.7)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.8)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.7)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.8)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertTrue(testMetrics.getValue("afterAccuracy1") > testMetrics.getValue("beforeAccuracy1"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("beforeAccuracy2"))
        assertTrue(testMetrics.getValue("afterAccuracy2") > testMetrics.getValue("afterAccuracy1"))
    }

    @Test
    fun exportImportWithRMSPropOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(RMSProp(), tempDir)

        assertTrue(testMetrics.getValue("trainAccuracy") > 0.3)
        assertTrue(testMetrics.getValue("beforeAccuracy1") > 0.3)
        assertTrue(testMetrics.getValue("afterAccuracy1") > 0.3)
        assertTrue(testMetrics.getValue("beforeAccuracy2") > 0.3)
        assertTrue(testMetrics.getValue("afterAccuracy2") > 0.3)
        assertEquals(testMetrics.getValue("trainAccuracy"), testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(testMetrics.getValue("beforeAccuracy2"), testMetrics.getValue("beforeAccuracy1"), EPS)
    }

    private fun trainingAndInferenceWithSpecificOptimizer(optimizer: Optimizer, tempDir: Path?): Map<String, Double> {
        val testMetrics = mutableMapOf<String, Double>()

        val (train, test) = mnist()

        val (newTrain, validation) = train.split(0.95)

        val lenet5 = Sequential.of(lenet5Layers)
        lenet5.use {
            it.compile(optimizer = optimizer, loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE
            )

            it.save(
                modelDirectory = tempDir!!.toFile(),
                saveOptimizerState = true,
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            testMetrics.put("trainAccuracy", accuracy!!)
        }

        val model = Sequential.loadDefaultModelConfiguration(tempDir!!.toFile())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            it.layers.filterIsInstance<Conv2D>().forEach(Layer::freeze)

            it.compile(
                optimizer = optimizer,
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.loadWeights(tempDir.toFile(), loadOptimizerState = true)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["beforeAccuracy1"] = accuracyBefore!!

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 1,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics.put("afterAccuracy1", accuracyAfterTraining!!)
        }

        val model2 = Sequential.loadDefaultModelConfiguration(tempDir.toFile())

        model2.use {
            it.compile(
                optimizer = optimizer,
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.loadWeights(tempDir.toFile(), loadOptimizerState = false)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["beforeAccuracy2"] = accuracyBefore!!

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 1,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["afterAccuracy2"] = accuracyAfterTraining!!
        }

        println("Test metrics: $testMetrics")
        return testMetrics.toMap()
    }
}
