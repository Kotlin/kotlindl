package examples.production

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
import datasets.Dataset
import datasets.handlers.AMOUNT_OF_CLASSES
import org.tensorflow.Tensor

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val kernelInitializer = GlorotNormal(SEED)
private val biasInitializer = GlorotUniform(SEED)

val lenet5 = Sequential.of(
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

fun getLabel(dataset: Dataset, imageId: Int): Int {
    val imageLabel = dataset.getY(imageId)
    return imageLabel.indexOf(imageLabel.max()!!)
}

fun mnistReshape(image: FloatArray): Tensor<*>? {
    val reshaped = Array(
        1
    ) { Array(28) { Array(28) { FloatArray(1) } } }
    for (i in image.indices) reshaped[0][i / 28][i % 28][0] = image[i]
    return Tensor.create(reshaped)
}