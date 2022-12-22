/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.integration

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.isTrainable
import org.jetbrains.kotlinx.dl.api.core.layer.weights
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.io.File

private const val pathToConfig = "inference/lenet/modelConfig.json"
private val realPathToConfig = TransferLearningTest::class.java.classLoader.getResource(pathToConfig).path.toString()

private const val pathToMismatchConfig = "inference/lenet/mismatchModelConfig.json"
private val realPathToMismatchConfig =
    TransferLearningTest::class.java.classLoader.getResource(pathToMismatchConfig).path.toString()


private const val pathToIncorrectConfig = "inference/lenet/unsupportedInitializers/modelConfig.json"
private val realPathToIncorrectConfig =
    TransferLearningTest::class.java.classLoader.getResource(pathToIncorrectConfig).path.toString()

private const val pathToConfigWithWrongJSON = "inference/lenet/wrongJson/model.json"
private val realPathToConfigWithWrongJSON =
    TransferLearningTest::class.java.classLoader.getResource(pathToConfigWithWrongJSON).path.toString()

private const val pathToWeights = "inference/lenet/mnist_weights_only.h5"
private val realPathToWeights = TransferLearningTest::class.java.classLoader.getResource(pathToWeights).path.toString()

private const val pathToConfigWithRegularizers = "inference/lenet/regularizers/modelConfig.json"
private val realPathToConfigRegularizers =
    TransferLearningTest::class.java.classLoader.getResource(pathToConfigWithRegularizers).path.toString()

private const val pathToWeightsRegularizers = "inference/lenet/regularizers/mnist_weights_only.h5"
private val realPathToWeightsRegularizers =
    TransferLearningTest::class.java.classLoader.getResource(pathToWeightsRegularizers).path.toString()

private const val pathToConfigWithDropout = "inference/lenet/dropout/modelConfig.json"
private val realPathToConfigDropout =
    TransferLearningTest::class.java.classLoader.getResource(pathToConfigWithDropout).path.toString()

private const val pathToWeightsDropout = "inference/lenet/dropout/weights.h5"
private val realPathToWeightsDropout =
    TransferLearningTest::class.java.classLoader.getResource(pathToWeightsDropout).path.toString()

class TransferLearningTest : IntegrationTest() {
    /** Loads configuration with default initializers for the last Dense layer from Keras. But Zeros initializer (default initializers for bias) is not supported yet. */
    /*@Test
    fun loadIncorrectSequentialJSONConfig() {
        val jsonConfigFile = File(realPathToIncorrectConfig)

        val exception =
            assertThrows(IllegalArgumentException::class.java) {
                Sequential.loadModelConfiguration(jsonConfigFile)
            }
        assertEquals(
            "Zeros is not supported yet!",
            exception.message
        )
    }*/

    /**
     * This test covers the case, when Python Keras user saves JSON as a String.
     *
     * ```
     * json_config = model.to_json()
     *    with open('keras-cifar-10/model.json', 'w') as f:
     *    json.dump(json_config, f)
     * ```
     */
    @Test
    fun loadSequentialJSONConfigWithUnsupportedJSONFormat() {
        val jsonConfigFile = File(realPathToConfigWithWrongJSON)

        val exception =
            assertThrows(IllegalArgumentException::class.java) {
                Sequential.loadModelConfiguration(jsonConfigFile)
            }
        assertEquals(
            "JSON file: model.json contains invalid JSON. The model configuration could not be loaded from this file.",
            exception.message
        )
    }


    @Test
    fun loadMismatchJSONConfigAndH5Weights() {
        val jsonConfig = File(realPathToMismatchConfig)
        val model = Sequential.loadModelConfiguration(jsonConfig)
        val modelDirectory = HdfFile(File(realPathToWeights))

        model.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        val exception =
            assertThrows(IllegalStateException::class.java) {
                model.loadWeights(modelDirectory)
            }
        assertEquals(
            "Weights for the loaded layer conv2d_12 are not found in .h5 file! \n" +
                    "h5 weight file contains weights for the following list of layers: [conv2d, conv2d_1, dense, dense_1, dense_2, flatten, max_pooling2d, max_pooling2d_1]\n" +
                    "Double-check your loaded configuration which contains layers with the following names: [input, conv2d_12, max_pooling2d_96, conv2d_1, max_pooling2d_1, flatten, dense, dense_1, dense_2].",
            exception.message
        )
    }

    @Test
    fun loadModelConfigFromKeras() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val flattenLayerName = "flatten"
        val conv2dLayerName = "conv2d"
        val conv2d1LayerName = "conv2d_1"
        val denseLayerName = "dense"
        val dense1LayerName = "dense_1"

        assertEquals(testModel.layers.size, 9)
        assertTrue(!testModel.getLayer(flattenLayerName).isTrainable)
        assertFalse(testModel.getLayer(flattenLayerName).hasActivation)
        assertTrue(testModel.getLayer(conv2dLayerName) is Conv2D)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).kernelInitializer is GlorotNormal)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).biasInitializer is GlorotUniform)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).padding == ConvPadding.SAME)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).activation == Activations.Relu)
        assertTrue(testModel.getLayer(conv2dLayerName).isTrainable)
        assertTrue(testModel.getLayer(conv2dLayerName).hasActivation)
        assertTrue((testModel.getLayer(conv2d1LayerName) as Conv2D).kernelInitializer is HeNormal)
        assertTrue((testModel.getLayer(conv2d1LayerName) as Conv2D).biasInitializer is HeUniform)
        assertTrue((testModel.getLayer(conv2d1LayerName) as Conv2D).padding == ConvPadding.SAME)
        assertTrue((testModel.getLayer(conv2d1LayerName) as Conv2D).activation == Activations.Relu)
        assertTrue((testModel.getLayer(denseLayerName) as Dense).kernelInitializer is LeCunNormal)
        assertTrue((testModel.getLayer(denseLayerName) as Dense).biasInitializer is LeCunUniform)
        assertTrue((testModel.getLayer(denseLayerName) as Dense).outputSize == 256)
        assertTrue((testModel.getLayer(denseLayerName) as Dense).activation == Activations.Relu)
        assertTrue((testModel.getLayer(dense1LayerName) as Dense).kernelInitializer is RandomNormal)
        assertTrue((testModel.getLayer(dense1LayerName) as Dense).biasInitializer is RandomUniform)
        assertTrue((testModel.getLayer(dense1LayerName) as Dense).outputSize == 84)
        assertTrue((testModel.getLayer(dense1LayerName) as Dense).activation == Activations.Relu)
        assertArrayEquals(testModel.inputLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    /** Weights are not loaded, but initialized via default initializers. */
    @Test
    fun loadModelConfigFromKerasAndTrain() {
        val jsonConfigFile = File(realPathToConfig)

        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val (train, test) = fashionMnist()

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.fit(
                dataset = train,
                validationRate = VALIDATION_RATE,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = VALIDATION_BATCH_SIZE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = VALIDATION_BATCH_SIZE).metrics[Metrics.ACCURACY]

            if (accuracy != null) {
                assertTrue(accuracy > 0.7)
            }
        }
    }

    /** Compilation is missed. */
    @Test
    fun loadModelConfigFromKerasAndMissCompilation() {
        val jsonConfigFile = File(realPathToConfig)

        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val (train, _) = fashionMnist()

        testModel.use {
            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        validationRate = VALIDATION_RATE,
                        epochs = EPOCHS,
                        trainBatchSize = TRAINING_BATCH_SIZE,
                        validationBatchSize = VALIDATION_BATCH_SIZE
                    )
                }
            assertEquals(
                "The model is not compiled yet. Compile the model to use this method.",
                exception.message
            )
        }
    }

    @Test
    fun loadWeights() {
        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)
        assertEquals(3400864L, hdfFile.size())

        val name = "conv2d"
        val kernelData = hdfFile.getDatasetByPath("/$name/$name/kernel:0").data as Array<Array<Array<FloatArray>>>
        val biasData = hdfFile.getDatasetByPath("/$name/$name/bias:0").data as FloatArray

        assertEquals(kernelData[0][0][0][0], 0.06445057f)
        assertEquals(biasData[15], -0.25060207f)
    }

    @Test
    fun loadModelConfigAndWeightsFromKeras() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val conv2DKernelWeights =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights[0][0][0][0], 0.06445057f)

            val conv2DKernelWeights1 =
                it.getLayer("conv2d_1").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights1[0][0][0][0], 0.027743129f)
        }
    }

    @Test
    fun loadModelConfigAndWeightsFromKerasWithRegularizers() {
        val jsonConfigFile = File(realPathToConfigRegularizers)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeightsRegularizers)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val conv2DKernelWeights =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06105182f, conv2DKernelWeights[0][0][0][0])

            val conv2DKernelWeights1 =
                it.getLayer("conv2d_1").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.032438572f, conv2DKernelWeights1[0][0][0][0])
        }
    }

    @Test
    fun loadModelConfigAndWeightsFromKerasWithDropout() {
        val jsonConfigFile = File(realPathToConfigDropout)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeightsDropout)
        val hdfFile = HdfFile(file)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val copy = it.copy()
            assertTrue(copy.layers.size == 11)
            copy.close()

            val conv2DKernelWeights =
                it.getLayer("conv2d_12").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.041764896f, conv2DKernelWeights[0][0][0][0])

            val conv2DKernelWeights1 =
                it.getLayer("conv2d_13").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.021932468f, conv2DKernelWeights1[0][0][0][0])
        }
    }

    @Test
    fun loadModelConfigAndWeightsFromKerasByTemplates() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            val kernelDataPathTemplate = "/%s/%s/kernel:0"
            val biasDataPathTemplate = "/%s/%s/bias:0"
            it.loadWeightsByPathTemplates(hdfFile, kernelDataPathTemplate, biasDataPathTemplate)

            val conv2DKernelWeights =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights[0][0][0][0], 0.06445057f)

            val conv2DKernelWeights1 =
                it.getLayer("conv2d_1").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights1[0][0][0][0], 0.027743129f)
        }
    }

    @Test
    fun loadModelConfigAndWeightsFromKerasByPaths() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            val weightPaths = listOf(
                LayerConvOrDensePaths(
                    "conv2d",
                    "/conv2d/conv2d/kernel:0",
                    "/conv2d/conv2d/bias:0"
                ),
                LayerConvOrDensePaths(
                    "conv2d_1",
                    "/conv2d_1/conv2d_1/kernel:0",
                    "/conv2d_1/conv2d_1/bias:0"
                ),
                LayerConvOrDensePaths(
                    "dense",
                    "/dense/dense/kernel:0",
                    "/dense/dense/bias:0"
                ),
                LayerConvOrDensePaths(
                    "dense_1",
                    "/dense_1/dense_1/kernel:0",
                    "/dense_1/dense_1/bias:0"
                ),
                LayerConvOrDensePaths(
                    "dense_2",
                    "/dense_2/dense_2/kernel:0",
                    "/dense_2/dense_2/bias:0"
                )
            )
            it.loadWeightsByPaths(hdfFile, weightPaths)

            val conv2DKernelWeights =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights[0][0][0][0], 0.06445057f)

            val conv2DKernelWeights1 =
                it.getLayer("conv2d_1").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeights1[0][0][0][0], 0.027743129f)
        }
    }

    @Test
    fun loadModelConfigAndWeightsTwiceFromKeras() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val exception =
                assertThrows(IllegalStateException::class.java) {
                    it.loadWeights(hdfFile)
                }
            assertEquals(
                "Model is already initialized.",
                exception.message
            )
        }
    }

    /** Simple transfer learning with additional training and without layers freezing. */
    @Test
    fun loadModelConfigAndWeightsFromKerasAndTrain() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = fashionMnist()

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyBefore != null) {
                assertTrue(accuracyBefore > 0.8)
            }

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 3,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyAfterTraining != null && accuracyBefore != null) {
                assertTrue(accuracyAfterTraining > accuracyBefore)
            }
        }

    }

    /** Simple transfer learning with additional training and Conv2D layers weights freezing. */
    @Test
    fun loadModelConfigAndWeightsFromKerasAndTrainDenseLayersOnly() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = fashionMnist()

        testModel.use {
            it.layers.filterIsInstance<Conv2D>().forEach(Layer::freeze)

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyBefore != null) {
                assertTrue(accuracyBefore > 0.8)
            }

            val conv2DKernelWeightsBeforeTraining =
                it.getLayer("conv2d").weights["conv2d_conv2d_kernel"] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsBeforeTraining[0][0][0][0], 0.06445057f)

            val denseDKernelWeightsBeforeTraining =
                it.getLayer("dense").weights["dense_dense_kernel"] as Array<FloatArray>
            assertEquals(denseDKernelWeightsBeforeTraining[0][0], 0.012644082f)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 3,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val conv2DKernelWeightsAfterTraining =
                it.getLayer("conv2d").weights["conv2d_conv2d_kernel"] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsAfterTraining[0][0][0][0], 0.06445057f)
            assertArrayEquals(conv2DKernelWeightsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining = it.getLayer("dense").weights["dense_dense_kernel"]
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyAfterTraining != null && accuracyBefore != null) {
                assertTrue(accuracyAfterTraining > accuracyBefore)
            }
        }
    }

    /**
     * Simple transfer learning with additional training and Conv2D layers weights freezing.
     *
     * NOTE: Dense weights are initialized via default initializers and trained from zero to hero.
     */
    @Test
    fun loadModelConfigAndWeightsPartiallyFromKerasAndTrainDenseLayersOnly() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = fashionMnist()

        testModel.use {
            val layerList = it.layers.filterIsInstance<Conv2D>()
            layerList.forEach(Layer::freeze)

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile, layerList)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyBefore != null) {
                assertTrue(accuracyBefore > 0.1) // Dense layers has no meaningful weights
            }

            val conv2DKernelWeighsBeforeTraining =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeighsBeforeTraining[0][0][0][0], 0.06445057f)
            val denseDKernelWeightsBeforeTraining =
                it.getLayer("dense").weights.values.toTypedArray()[0] as Array<FloatArray>
            assertEquals(denseDKernelWeightsBeforeTraining[0][0], 0.008463251f)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 4,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val conv2DKernelWeightsAfterTraining =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsAfterTraining[0][0][0][0], 0.06445057f)
            assertArrayEquals(conv2DKernelWeighsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining =
                it.getLayer("dense").weights.values.toTypedArray()[0] as Array<FloatArray>
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyAfterTraining != null && accuracyBefore != null) {
                assertTrue(accuracyAfterTraining > accuracyBefore)
            }
        }
    }

    /**
     * Simple transfer learning with additional training and Conv2D layers weights freezing.
     *
     * NOTE: Dense weights are initialized via default initializers and trained from zero to hero.
     */
    @Test
    fun loadModelConfigAndWeightsPartiallyByLayersListFromKerasAndTrainDenseLayersOnly() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = Sequential.loadModelConfiguration(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = fashionMnist()

        testModel.use {
            it.layers.filterIsInstance<Conv2D>().forEach(Layer::freeze)

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeightsForFrozenLayers(hdfFile)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyBefore != null) {
                assertTrue(accuracyBefore > 0.1) // Dense layers has no meaningful weights
            }

            val conv2DKernelWeighsBeforeTraining =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeighsBeforeTraining[0][0][0][0], 0.06445057f)
            val denseDKernelWeightsBeforeTraining =
                it.getLayer("dense").weights.values.toTypedArray()[0] as Array<FloatArray>
            assertEquals(denseDKernelWeightsBeforeTraining[0][0], 0.008463251f)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 4,
                trainBatchSize = 1000,
                validationBatchSize = 100
            )

            val conv2DKernelWeightsAfterTraining =
                it.getLayer("conv2d").weights.values.toTypedArray()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsAfterTraining[0][0][0][0], 0.06445057f)
            assertArrayEquals(conv2DKernelWeighsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining =
                it.getLayer("dense").weights.values.toTypedArray()[0] as Array<FloatArray>
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            if (accuracyAfterTraining != null && accuracyBefore != null) {
                assertTrue(accuracyAfterTraining > accuracyBefore)
            }
        }
    }
}
