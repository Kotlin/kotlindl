package api.keras.integration

import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.activations.Activations
import api.keras.dataset.Dataset
import api.keras.initializers.GlorotNormal
import api.keras.initializers.GlorotUniform
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

class TransferLearningTest : IntegrationTest() {
    @Test
    fun loadSequentialJSONConfig() {
        val pathToConfig = "models/mnist/lenet/model_with_glorot_normal_init.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        assertEquals(testModel.layers.size, 8)
        assertTrue(testModel.getLayer("flatten_7").isTrainable)
        assertFalse(testModel.getLayer("flatten_7").hasActivation())
        assertTrue(testModel.getLayer("conv2d_14") is Conv2D)
        assertTrue((testModel.getLayer("conv2d_14") as Conv2D).kernelInitializer is GlorotNormal)
        assertTrue((testModel.getLayer("conv2d_14") as Conv2D).biasInitializer is GlorotNormal)
        assertTrue((testModel.getLayer("conv2d_14") as Conv2D).padding == ConvPadding.SAME)
        assertTrue((testModel.getLayer("conv2d_14") as Conv2D).activation == Activations.Relu)
        assertTrue(testModel.getLayer("conv2d_15") is Conv2D)
        assertTrue(testModel.getLayer("conv2d_14").isTrainable)
        assertTrue(testModel.getLayer("conv2d_15").hasActivation())

        assertArrayEquals(testModel.firstLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    @Test
    fun loadSequentialJSONConfigAndTrain() {
        val pathToConfig = "models/mnist/lenet/model_with_glorot_normal_init.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
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
    // NOTE: LeNet model here has bad initializers that are in-effective for training from zero to hero.
    // TODO: Re-run LeNet in Python with best initializers and remove Sequential tests above
    fun loadLeNetJSONConfig() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)


        assertEquals(testModel.layers.size, 8)
        assertTrue(testModel.getLayer("flatten_4").isTrainable)
        assertFalse(testModel.getLayer("flatten_4").hasActivation())
        assertEquals(testModel.getLayer("conv2d_8")::class, Conv2D::class)
        assertEquals((testModel.getLayer("conv2d_8") as Conv2D).kernelInitializer::class, GlorotUniform::class)
        assertEquals((testModel.getLayer("conv2d_8") as Conv2D).biasInitializer::class, GlorotNormal::class)
        assertEquals((testModel.getLayer("conv2d_8") as Conv2D).padding, ConvPadding.SAME)
        assertEquals((testModel.getLayer("conv2d_8") as Conv2D).activation, Activations.Relu)
        assertEquals(testModel.getLayer("conv2d_9")::class, Conv2D::class)
        assertTrue(testModel.getLayer("conv2d_8").isTrainable)
        assertTrue(testModel.getLayer("conv2d_9").hasActivation())

        assertArrayEquals(testModel.firstLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    @Test
    fun loadLeNetJSONConfigAndTrain() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
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

            assertEquals(0.10000000149011612, accuracy!!, EPS)
        }
    }

    @Test
    fun loadWeights() {
        val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
        val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)
        assertEquals(hdfFile.size(), 3400872L)


        val name = "conv2d_8"
        val kernelData = hdfFile.getDatasetByPath("/$name/$name/kernel:0").data as Array<Array<Array<FloatArray>>>
        val biasData = hdfFile.getDatasetByPath("/$name/$name/bias:0").data as FloatArray

        assertEquals(0.06366586f, kernelData[0][0][0][0])
        assertEquals(0.0022679337f, biasData[15])
    }

    @Test
    fun loadWeightsAndJSONConfig() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
        val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
        val file = File(realPathToWeights)
        val hdfFile = HdfFile(file)

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.loadWeights(hdfFile)

            val conv2DKernelWeights = it.getLayer("conv2d_8").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06366586f, conv2DKernelWeights[0][0][0][0])
        }
    }

    @Test
    // Simplest transfer learning without freezing
    fun loadWeightsAndJSONConfigAndTrain() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
        val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
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

            assertEquals(accuracyBefore!!, 0.8286998271942139, EPS)

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

            assertEquals(0.8796000480651855, accuracyAfterTraining!!, EPS)
        }

    }

    @Test
    // Simplest transfer learning with Conv2D layers weights freezing
    fun loadWeightsAndJSONConfigAndTrainDenseLayers() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
        val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
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
                it.getLayer("conv2d_8").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsBeforeTraining[0][0][0][0], 0.06366586f)
            val denseDKernelWeightsBeforeTraining = it.getLayer("dense_12").getWeights()[0] as Array<FloatArray>
            assertEquals(denseDKernelWeightsBeforeTraining[0][0], -2.7934683E-4f)

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
                it.getLayer("conv2d_8").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeightsAfterTraining[0][0][0][0], 0.06366586f)
            assertArrayEquals(conv2DKernelWeightsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining = it.getLayer("dense_12").getWeights()[0] as Array<FloatArray>
            // NOTE: dense weights could be different from time to time but in short range
            assertEquals(denseDKernelWeightsAfterTraining[0][0].toDouble(), -0.0030897409, EPS)
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.869799792766571, accuracyAfterTraining!!, EPS)
        }
    }


    @Test
    // Simplest transfer learning with loading and freezing Conv2D weights, Dense weights are initialized and trained from zero to hero
    fun loadWeightsPartiallyAndJSONConfigAndTrainDenseLayers() {
        val pathToConfig = "models/mnist/lenet/lenetMdl.json"
        val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = buildModelByJSONConfig(jsonConfigFile)

        val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
        val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
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
                it.getLayer("conv2d_8").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(conv2DKernelWeighsBeforeTraining[0][0][0][0], 0.06366586f)
            val denseDKernelWeightsBeforeTraining = it.getLayer("dense_12").getWeights()[0] as Array<FloatArray>
            assertEquals(denseDKernelWeightsBeforeTraining[0][0], 0.03821275f) // Random initialization

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
                it.getLayer("conv2d_8").getWeights()[0] as Array<Array<Array<FloatArray>>>
            assertEquals(0.06366586f, conv2DKernelWeightsAfterTraining[0][0][0][0])
            assertArrayEquals(conv2DKernelWeighsBeforeTraining, conv2DKernelWeightsAfterTraining)

            val denseDKernelWeightsAfterTraining = it.getLayer("dense_12").getWeights()[0] as Array<FloatArray>
            // NOTE: dense weights could be different from time to time but in short range
            assertEquals(denseDKernelWeightsAfterTraining[0][0].toDouble(), 0.04153359681367874, EPS)
            assertFalse(denseDKernelWeightsBeforeTraining.contentEquals(denseDKernelWeightsAfterTraining))

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            assertEquals(0.5108000040054321, accuracyAfterTraining!!, EPS)
        }
    }
}