Kotof is Keras API at the top of TensorFlow 1.15 and its Java API.

The implementation is inspired by https://github.com/dhruvrajan and its Java and Scala Keras APIs 
in https://github.com/dhruvrajan/tensorflow-keras-java and https://github.com/dhruvrajan/tensorflow-keras-scala.

Classic LeNet-5 with minor changes looks so Keras, isn't it?

```kotlin
private val model = Sequential.of<Float>(
    Input(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 512,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    )
)

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)

    model.use {
        it.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(trainDataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, isDebugMode = true)

        val accuracy = it.evaluate(testDataset = test, metric = Metrics.ACCURACY)

        println("Accuracy: $accuracy")
    }
}
```

**Work on GPU**

To enable the training and inference on GPU developer should read this document https://www.tensorflow.org/install/gpu and install the CUDA framework to bring the calculations on GPU device.

First of all, it's possible for NVIDIA devices only.

After that, you need to build with the next dependencies

* _compile 'org.tensorflow:libtensorflow:1.15.0'_

* _compile 'org.tensorflow:libtensorflow_jni_gpu:1.15.0'_

For TF 1.15 (the TF version is under the hood) the next bunch of versions is required
  * Windows:  CUDA cuda_10.0.130_411.31_win10 and cudnn-10.0 (https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/cudnn-10.0-windows10-x64-v7.6.3.30.zip) and C++ redistributable parts (2015, https://www.microsoft.com/en-us/download/details.aspx?id=48145) 