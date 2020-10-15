package api.core.models

import api.core.Sequential
import api.core.activation.Activations
import api.core.exception.RepeatableLayerNameException
import api.core.initializer.Constant
import api.core.initializer.HeNormal
import api.core.initializer.Zeros
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.Losses
import api.core.metric.Accuracy
import api.core.optimizer.Adam
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val AMOUNT_OF_CLASSES = 10


internal class SequentialModelTest {
    private val correctTestModel = Sequential.of(
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
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_2"
        )
    )

    @Test
    fun buildModel() {
        assertEquals(correctTestModel.layers.size, 7)
        assertTrue(correctTestModel.getLayer("conv2d_1") is Conv2D)
        assertTrue(correctTestModel.getLayer("conv2d_2") is Conv2D)
        assertTrue(correctTestModel.getLayer("conv2d_1").isTrainable)
        assertTrue(correctTestModel.getLayer("conv2d_1").hasActivation())
        assertTrue(correctTestModel.getLayer("flatten_1").isTrainable)
        assertFalse(correctTestModel.getLayer("flatten_1").hasActivation())
        assertArrayEquals(correctTestModel.inputLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    @Test
    fun summary() {
        correctTestModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())
            val layerDescriptions = it.summary()
            assertTrue(layerDescriptions[0].contentEquals("conv2d_1(Conv2D)             [-1, 28, 28, 32]          832"))
        }
    }

    @Test
    fun compilation() {
        val exception =
            assertThrows(UninitializedPropertyAccessException::class.java) { correctTestModel.layers[0].getParams() }
        assertEquals(
            "lateinit property kernelShape has not been initialized",
            exception.message
        )

        assertFalse(correctTestModel.isModelCompiled)

        correctTestModel.use {
            it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Accuracy())
            assertTrue(correctTestModel.isModelCompiled)

            assertEquals(it.layers[0].getParams(), 832)
            assertEquals(it.layers[1].getParams(), 0)
            assertEquals(it.layers[2].getParams(), 51264)
            assertEquals(it.layers[3].getParams(), 0)
            assertEquals(it.layers[4].getParams(), 0)
            assertEquals(it.layers[5].getParams(), 1606144)
            assertEquals(it.layers[6].getParams(), 5130)

            assertArrayEquals(it.layers[0].outputShape, longArrayOf(-1, 28, 28, 32))
            assertArrayEquals(it.layers[1].outputShape, longArrayOf(-1, 14, 14, 32))
            assertArrayEquals(it.layers[2].outputShape, longArrayOf(-1, 14, 14, 64))
            assertArrayEquals(it.layers[3].outputShape, longArrayOf(-1, 7, 7, 64))
            assertArrayEquals(it.layers[4].outputShape, longArrayOf(3136))
            assertArrayEquals(it.layers[5].outputShape, longArrayOf(512))
            assertArrayEquals(it.layers[6].outputShape, longArrayOf(10))
        }
    }

    @Test
    fun repeatableNamesFails() {
        val exception = assertThrows(RepeatableLayerNameException::class.java) {
            Sequential.of(
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
                Conv2D(
                    filters = 64,
                    kernelSize = longArrayOf(5, 5),
                    strides = longArrayOf(1, 1, 1, 1),
                    activation = Activations.Relu,
                    kernelInitializer = HeNormal(SEED),
                    biasInitializer = Zeros(),
                    padding = ConvPadding.SAME,
                    name = "conv2d_1"
                )
            )
        }

        assertEquals(
            "The layer name conv2d_1 is used in previous layers. The layer name should be unique.",
            exception.message
        )
    }

    @Test
    fun namesGeneration() {
        val model = Sequential.of(
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
                padding = ConvPadding.SAME
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1)
            ),
            Flatten(),
            Dense(
                outputSize = AMOUNT_OF_CLASSES,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f)
            )
        )

        assertEquals(model.layers[0].name, "conv2d_1")
        assertEquals(model.layers[1].name, "maxpool2d_2")
        assertEquals(model.layers[2].name, "flatten_3")
        assertEquals(model.layers[3].name, "dense_4")
    }
}