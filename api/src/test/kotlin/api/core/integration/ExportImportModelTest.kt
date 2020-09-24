package api.core.integration

import api.core.ModelFormat
import api.core.ModelWritingMode
import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.GlorotNormal
import api.core.initializer.GlorotUniform
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.*
import datasets.Dataset
import datasets.handlers.*
import org.junit.Ignore
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

private const val EPS = 0.1
private const val SEED = 12L
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000
private const val AMOUNT_OF_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private val kernelInitializer = GlorotNormal(SEED)

private val biasInitializer = GlorotUniform(SEED)

class ExportImportModelTest {
    private val lenet5Layers = listOf(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS,
            name = "input_0"
        ),
        Conv2D(
            filters = 32,
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
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
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
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
    fun exportImport(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        val (newTrain, validation) = train.split(0.95)

        lenet5.use {
            it.compile(optimizer = SGD(learningRate = 0.05f), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.summary()

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = true
            )

            it.save(
                pathToModelDirectory = tempDir.toString(),
                modelFormat = ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES,
                modelWritingMode = ModelWritingMode.OVERRIDE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            assertEquals(0.7603000402450562, accuracy!!, EPS)
        }

        val model = Sequential.load(tempDir.toString())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = RMSProp(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.summary()

            it.loadVariablesFromTxtFiles(tempDir.toString())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.7603000402450562, accuracyBefore!!, EPS)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = false,
                isWeightsInitRequired = false // for transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.8234000205993652, accuracyAfterTraining!!, EPS)
        }
    }

    @Test
    fun exportImportWithAdamOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Adam(), tempDir)
        assertEquals(0.9406000375747681, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.9406002759933472, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.9580999612808228, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.9406002759933472, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.9708000421524048, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    fun exportImportWithAdaDeltaOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(AdaDelta(), tempDir)
        assertEquals(0.6644999980926514, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.6645000576972961, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.7969000935554504, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.6645000576972961, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.7883001565933228, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithAdaGradOptimizerInternalState(@TempDir tempDir: Path?) {
        val testMetrics =
            trainingAndInferenceWithSpecificOptimizer(AdaGrad(clipGradient = ClipGradientByValue(0.01f)), tempDir)
        assertEquals(0.1, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithAdaGradDAOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(AdaGradDA(), tempDir)
        assertEquals(0.11349999904632568, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithAdamaxOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Adamax(), tempDir)
        assertEquals(0.1, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithFtrlOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Ftrl(), tempDir)
        assertEquals(0.1, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithMomentumOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Momentum(), tempDir)
        assertEquals(0.1, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithRMSPropOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(RMSProp(), tempDir)
        assertEquals(0.1, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.1, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.1, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    private fun trainingAndInferenceWithSpecificOptimizer(optimizer: Optimizer, tempDir: Path?): Map<String, Double> {
        val testMetrics = mutableMapOf<String, Double>()

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        val (newTrain, validation) = train.split(0.95)

        val lenet5 = Sequential.of(lenet5Layers)
        lenet5.use {
            it.compile(optimizer = optimizer, loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = true
            )

            it.save(
                pathToModelDirectory = tempDir.toString(),
                saveOptimizerState = true,
                modelFormat = ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES,
                modelWritingMode = ModelWritingMode.OVERRIDE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            testMetrics.put("trainAccuracy", accuracy!!)
        }

        val model = Sequential.load(tempDir.toString())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = optimizer,
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.loadVariablesFromTxtFiles(tempDir.toString(), loadOptimizerState = true)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["beforeAccuracy1"] = accuracyBefore!!

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 1,
                trainBatchSize = 1000,
                validationBatchSize = 100,
                verbose = true,
                isWeightsInitRequired = false, // for transfer learning
                isOptimizerInitRequired = false // for optimizer transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics.put("afterAccuracy1", accuracyAfterTraining!!)
        }

        val model2 = Sequential.load(tempDir.toString())

        model2.use {
            it.compile(
                optimizer = optimizer,
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.loadVariablesFromTxtFiles(tempDir.toString(), loadOptimizerState = false)

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["beforeAccuracy2"] = accuracyBefore!!

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = 1,
                trainBatchSize = 1000,
                validationBatchSize = 100,
                verbose = true,
                isWeightsInitRequired = false, // for transfer learning
                isOptimizerInitRequired = true // for optimizer transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics["afterAccuracy2"] = accuracyAfterTraining!!
        }

        return testMetrics.toMap()
    }
}