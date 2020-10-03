package api.core.integration

import api.core.SavingFormat
import api.core.Sequential
import api.core.WrintingMode
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.initializer.HeUniform
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.*
import datasets.Dataset
import datasets.handlers.*
import org.junit.Ignore
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
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
    fun exportImportWithValidation(@TempDir tempDir: Path?) {
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
            it.compile(optimizer = SGD(learningRate = 0.05f), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.summary()

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            assertEquals(0.766700029373169, accuracy!!, EPS)

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )
        }

        // TODO: refactor to ./modelConfig.json
        val model = Sequential.loadModelConfiguration(tempDir.toFile())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = RMSProp(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.summary()

            it.loadWeights(tempDir.toFile())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.766700029373169, accuracyBefore!!, EPS)

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
            assertEquals(0.8662000894546509, accuracyAfterTraining!!, EPS)
        }
    }

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

        lenet5.use {
            it.compile(optimizer = SGD(learningRate = 0.05f), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.summary()

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE,
                verbose = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            assertEquals(0.6370999813079834, accuracy!!, EPS)

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )
        }

        val model = Sequential.loadModelConfiguration(File(tempDir.toFile().absolutePath + "/modelConfig.json"))

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = RMSProp(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.summary()

            it.loadWeights(tempDir.toFile())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.6370999813079834, accuracyBefore!!, EPS)

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
            assertEquals(0.8650000691413879, accuracyAfterTraining!!, EPS)
        }
    }

    @Test
    fun exportImportWithAdamOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(Adam(), tempDir)
        assertEquals(0.912600040435791, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.9126001596450806, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.9462999105453491, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.9126001596450806, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.9514999389648438, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    fun exportImportWithAdaDeltaOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(AdaDelta(), tempDir)
        assertEquals(0.6964999437332153, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.6965000033378601, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.842700183391571, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.6965000033378601, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.8596002459526062, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
    @Ignore
    fun exportImportWithAdaGradOptimizerInternalState(@TempDir tempDir: Path?) {
        val testMetrics =
            trainingAndInferenceWithSpecificOptimizer(AdaGrad(clipGradient = ClipGradientByValue(0.01f)), tempDir)
        assertEquals(0.3929999768733978, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.3930000960826874, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.8846998810768127, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.3930000960826874, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.9439000487327576, testMetrics.getValue("afterAccuracy2"), EPS)
    }

    @Test
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
    fun exportImportWithRMSPropOptimizerInternalState(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)
        val testMetrics = trainingAndInferenceWithSpecificOptimizer(RMSProp(), tempDir)
        assertEquals(0.6583000421524048, testMetrics.getValue("trainAccuracy"), EPS)
        assertEquals(0.6582998633384705, testMetrics.getValue("beforeAccuracy1"), EPS)
        assertEquals(0.5664000511169434, testMetrics.getValue("afterAccuracy1"), EPS)
        assertEquals(0.6582998633384705, testMetrics.getValue("beforeAccuracy2"), EPS)
        assertEquals(0.616899847984314, testMetrics.getValue("afterAccuracy2"), EPS)
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
            it.compile(optimizer = optimizer, loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = true
            )

            it.save(
                modelDirectory = tempDir!!.toFile(),
                saveOptimizerState = true,
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            testMetrics.put("trainAccuracy", accuracy!!)
        }

        // TODO: refactor to ./modelConfig.json
        val model = Sequential.loadModelConfiguration(tempDir!!.toFile())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

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
                validationBatchSize = 100,
                verbose = true,
                isWeightsInitRequired = false, // for transfer learning
                isOptimizerInitRequired = false // for optimizer transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            testMetrics.put("afterAccuracy1", accuracyAfterTraining!!)
        }

        // TODO: refactor to ./modelConfig.json
        val model2 = Sequential.loadModelConfiguration(tempDir.toFile())

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