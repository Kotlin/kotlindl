package examples.production

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.GlorotNormal
import api.keras.initializers.HeNormal
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import examples.keras.mnist.util.AMOUNT_OF_CLASSES

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val kernelInitializer = HeNormal<Float>(SEED)

private val biasInitializer = GlorotNormal<Float>(SEED)

val lenet5 = Sequential.of<Float>(
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
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "1"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "2"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "3"
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "4"
    ),
    Dense(
        outputSize = AMOUNT_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "5"
    )
)

fun getLabel(dataset: ImageDataset, imageId: Int): Int {
    val trainImageLabel = dataset.getImageLabel(imageId)
    return trainImageLabel.indexOf(trainImageLabel.max()!!)
}