package api.keras.integration

import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.activations.Activations
import api.keras.dataset.Dataset
import api.keras.initializers.*
import api.keras.layers.Dense
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*
import io.jhdf.HdfFile
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.io.File

private const val pathToConfig = "inference/lenet/modelConfig.json"
private val realPathToConfig = TransferLearningTest::class.java.classLoader.getResource(pathToConfig).path.toString()

private const val pathToWeights = "inference/lenet/mnist_weights_only.h5"
private val realPathToWeights = TransferLearningTest::class.java.classLoader.getResource(pathToWeights).path.toString()

class TransferLearningTest : IntegrationTest() {
    @Test
    fun loadSequentialJSONConfig() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val flattenLayerName = "flatten"
        val conv2dLayerName = "conv2d"
        val conv2d1LayerName = "conv2d_1"
        val denseLayerName = "dense"
        val dense1LayerName = "dense_1"

        assertEquals(testModel.layers.size, 8)
        assertTrue(testModel.getLayer(flattenLayerName).isTrainable)
        assertFalse(testModel.getLayer(flattenLayerName).hasActivation())
        assertTrue(testModel.getLayer(conv2dLayerName) is Conv2D)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).kernelInitializer is GlorotNormal)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).biasInitializer is GlorotUniform)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).padding == ConvPadding.SAME)
        assertTrue((testModel.getLayer(conv2dLayerName) as Conv2D).activation == Activations.Relu)
        assertTrue(testModel.getLayer(conv2dLayerName).isTrainable)
        assertTrue(testModel.getLayer(conv2dLayerName).hasActivation())
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
        assertArrayEquals(testModel.firstLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    @Test
    fun loadSequentialJSONConfigAndTrain() {
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            FASHION_TRAIN_IMAGES_ARCHIVE,
            FASHION_TRAIN_LABELS_ARCHIVE,
            FASHION_TEST_IMAGES_ARCHIVE,
            FASHION_TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractFashionImages,
            ::extractFashionLabels
        )

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.fit(
                dataset = train,
                validationRate = VALIDATION_RATE,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = VALIDATION_BATCH_SIZE,
                verbose = true,
                isWeightsInitRequired = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = VALIDATION_BATCH_SIZE).metrics[Metrics.ACCURACY]

            assertEquals(0.788399875164032, accuracy!!, EPS)
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

        assertEquals(0.06169984f, kernelData[0][0][0][0])
        assertEquals(-0.25060207f, biasData[15])
    }

    @Test
    fun loadWeightsAndJSONConfig() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val conv2DKernelWeights = it.getLayer("conv2d").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06169984f, conv2DKernelWeights[0][0][0][0])

            val conv2DKernelWeights1 = it.getLayer("conv2d_1").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.04101113f, conv2DKernelWeights1[0][0][0][0])
        }
    }

    @Test
    // Simplest transfer learning without freezing
    fun loadWeightsAndJSONConfigAndTrain() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            FASHION_TRAIN_IMAGES_ARCHIVE,
            FASHION_TRAIN_LABELS_ARCHIVE,
            FASHION_TEST_IMAGES_ARCHIVE,
            FASHION_TEST_LABELS_ARCHIVE,
            datasets.AMOUNT_OF_CLASSES,
            ::extractFashionImages,
            ::extractFashionLabels
        )

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.8286998271942139, accuracyBefore!!, EPS)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 3,
                trainBatchSize = 1000,
                validationBatchSize = 100,
                verbose = false,
                isWeightsInitRequired = false // for transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.8623998761177063, accuracyAfterTraining!!, EPS)
        }

    }

    @Test
    // Simplest transfer learning with Conv2D layers weights freezing
    fun loadWeightsAndJSONConfigAndTrainDenseLayers() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            FASHION_TRAIN_IMAGES_ARCHIVE,
            FASHION_TRAIN_LABELS_ARCHIVE,
            FASHION_TEST_IMAGES_ARCHIVE,
            FASHION_TEST_LABELS_ARCHIVE,
            datasets.AMOUNT_OF_CLASSES,
            ::extractFashionImages,
            ::extractFashionLabels
        )

        testModel.use {
            for (layer in it.layers) {
                if (layer is Conv2D)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.8286998271942139, accuracyBefore!!, EPS)

            val conv2DKernelWeightsBeforeTraining =
                it.getLayer("conv2d").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06169984f, conv2DKernelWeightsBeforeTraining[0][0][0][0])

            val denseDKernelWeightsBeforeTraining = it.getLayer("dense").getWeights()[0] as Array<FloatArray>
            assertEquals(0.007553822f, denseDKernelWeightsBeforeTraining[0][0])

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 3,
                trainBatchSize = 1000,
                validationBatchSize = 100,
                verbose = false,
                isWeightsInitRequired = false // for transfer learning
            )

            val conv2DKernelWeightsAfterTraining =
                it.getLayer("conv2d").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06169984f, conv2DKernelWeightsAfterTraining[0][0][0][0])
            assertArrayEquals(conv2DKernelWeightsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining = it.getLayer("dense").getWeights()[0] as Array<FloatArray>
            // NOTE: dense weights could be different from time to time but in short range
            assertEquals(0.009714942425489426, denseDKernelWeightsAfterTraining[0][0].toDouble(), EPS)
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.8513997793197632, accuracyAfterTraining!!, EPS)
        }
    }

    @Test
    // Simplest transfer learning with loading and freezing Conv2D weights, Dense weights are initialized and trained from zero to hero
    fun loadWeightsPartiallyAndJSONConfigAndTrainDenseLayers() {
        val jsonConfigFile = File(realPathToConfig)
        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            FASHION_TRAIN_IMAGES_ARCHIVE,
            FASHION_TRAIN_LABELS_ARCHIVE,
            FASHION_TEST_IMAGES_ARCHIVE,
            FASHION_TEST_LABELS_ARCHIVE,
            datasets.AMOUNT_OF_CLASSES,
            ::extractFashionImages,
            ::extractFashionLabels
        )

        testModel.use {
            val layerList = mutableListOf<Layer>()

            for (layer in it.layers) {
                if (layer is Conv2D) {
                    layer.isTrainable = false
                    layerList.add(layer)
                }
            }

            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile, layerList)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.10000000149011612, accuracyBefore!!, EPS) // Dense layers has no meaningful weights

            val conv2DKernelWeighsBeforeTraining =
                it.getLayer("conv2d").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06169984f, conv2DKernelWeighsBeforeTraining[0][0][0][0])
            val denseDKernelWeightsBeforeTraining = it.getLayer("dense").getWeights()[0] as Array<FloatArray>
            assertEquals(0.008463251f, denseDKernelWeightsBeforeTraining[0][0])

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 4,
                trainBatchSize = 1000,
                validationBatchSize = 100,
                verbose = false,
                isWeightsInitRequired = false // for transfer learning
            )

            val conv2DKernelWeightsAfterTraining =
                it.getLayer("conv2d").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06169984f, conv2DKernelWeightsAfterTraining[0][0][0][0])
            assertArrayEquals(conv2DKernelWeighsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining = it.getLayer("dense").getWeights()[0] as Array<FloatArray>
            // NOTE: dense weights could be different from time to time but in short range
            assertEquals(0.004773239139467478, denseDKernelWeightsAfterTraining[0][0].toDouble(), EPS)
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.8484998345375061, accuracyAfterTraining!!, 0.4) // flaky behavior
        }
    }
}