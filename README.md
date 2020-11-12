KotlinDL is a high-level Deep Learning API written in Kotlin and inspired by [Keras](https://keras.io). Under the 
 hood it is using TensorFlow Java API. 
KotlinDL offers simple APIs for training deep learning models from scratch, importing existing Keras models 
for inference, and leveraging transfer learning for tweaking existing pre-trained models to your tasks. 

This project aims to make Deep Learning easier for JVM developers, and to simplify deploying deep learning models
 in JVM production environments. 

```kotlin
private val lenet5Classic = Sequential.of<Float>(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 6,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.VALID
    ),
    Conv2D(
        filters = 16,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.VALID
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Tanh,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = AMOUNT_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = GlorotNormal(SEED),
        biasInitializer = Constant(0.1f)
    )
)

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    lenet5Classic.use {
        it.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.summary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}
```

## Table of Contents

- [TensorFlow Engine](#tensorflow-engine)
- [Limitations](#limitations)
- [How to configure KotlinDL in your project](#how-to-configure-kotlindl-in-your-project)
- [Working with KotlinDL in Jupyter Notebook](#working-with-kotlindl-in-jupyter-notebook)
- [Examples and tutorials](#examples-and-tutorials)
- [Running KotlinDL on GPU](#running-kotlindl-on-gpu)
- [Reporting issues/Support](#reporting-issuessupport)
- [Code of Conduct](#code-of-conduct)
- [License](#license)


## TensorFlow Engine
KotlinDL is built on top of TensorFlow 1.15 Java API. The Java API for TensorFlow 2.+ has recently had first public
 release, and this project will be switching to it in the nearest future. This, however, does not affect the high-level 
 API. 

## Limitations
Currently, only a limited set of deep learning architectures is supported. Here's the list of available layers: 
- Input()
- Flatten()
- Dense()
- Dropout()
- Conv2D()
- MaxPool2D()
- AvgPool2D()   

KotlinDL supports model inference in JVM backend applications, Android support is coming in later releases.  

## How to configure KotlinDL in your project
To use KotlinDL in your project, you need to add the following dependency to your `build.gradle` file:
```kotlin
   repositories {
       maven {
           url  "https://kotlin.bintray.com/kotlin-datascience"
       }
   }
   
   dependencies {
       implementation 'org.jetbrains.kotlin-deeplearning:api:[KOTLIN-DL-VERSION]'
   }
```
For more details, as well as for `pom.xml` and `build.gradle.kts` examples, please refer to the [Quick Start Guide](docs/quick_start_guide.md).

## Working with KotlinDL in Jupyter Notebook
You can work with KotlinDL interactively in Jupyter Notebook with Kotlin kernel. To do so, add the following dependency 
in your notebook: 

```
   @file:Repository("https://kotlin.bintray.com/kotlin-datascience")
   @file:DependsOn("org.jetbrains.kotlin-deeplearning:api:[KOTLIN-DL-VERSION]")
```

For more details on how to install Jupyter Notebook and add Kotlin kernel, check out the [Quick Start Guide](docs/quick_start_guide.md).

## Examples and tutorials
You do not need to have any prior deep learning experience to start using KotlinDL. We are working on including extensive 
documentation to help you get started. At this point, feel free to check out the following tutorials:
- [Quick Start Guide](docs/quick_start_guide.md) 
- [Creating your first neural network](docs/create_your_first_nn.md)
- [Training a model](docs/training_a_model.md)
- [Running inference with a trained model](docs/loading_trained_model_for_inference.md)
- [Importing a Keras model](docs/importing_keras_model.md) 

For more inspiration, take a look at the [code examples](examples) in this repo.

## Running KotlinDL on GPU

To enable the training and inference on GPU, please read this [TensorFlow GPU Support page](https://www.tensorflow.org/install/gpu)
  and install the CUDA framework to enable calculations on a GPU device.

Note that only NVIDIA devices are supported.

You will also need to add the following dependencies in your project if you wish to leverage GPU: 

```
* _compile 'org.tensorflow:libtensorflow:1.15.0'_

* _compile 'org.tensorflow:libtensorflow_jni_gpu:1.15.0'_
```

On Windows the following distributions are required:
- CUDA cuda_10.0.130_411.31_win10
- [cudnn-10.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/cudnn-10.0-windows10-x64-v7.6.3.30.zip)
- [C++ redistributable parts](https://www.microsoft.com/en-us/download/details.aspx?id=48145) 

## Reporting issues/Support

Please use [GitHub issues](https://github.com/JetBrains/KotlinDL/issues) for filing feature requests and bug reports. 
You are also welcome to join [#deeplearning channel](https://kotlinlang.slack.com/archives/C01DZU7PW73) in the Kotlin Slack.

## Code of Conduct
This project and the corresponding community is governed by the [JetBrains Open Source and Community Code of Conduct](https://confluence.jetbrains.com/display/ALL/JetBrains+Open+Source+and+Community+Code+of+Conduct). Please make sure you read it. 

## License
KotlinDL is licensed under the [Apache 2.0 License](LICENSE).


