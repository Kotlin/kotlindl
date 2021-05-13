# 0.2.0 (01/05/2021)
Features:
* Added [support for Functional API](https://github.com/JetBrains/KotlinDL/issues/23)
* Added [BatchNorm layer](https://github.com/JetBrains/KotlinDL/issues/34) for inference
* Added [GlobalAveragePooling2D layer](https://github.com/JetBrains/KotlinDL/issues/38)
* Added [7 Merge layers](https://github.com/JetBrains/KotlinDL/issues/37) 
(Add, Average, Concatenate, Maximum, Minimum, Multiply, Subtract)
* Added [Activation layer](https://github.com/JetBrains/KotlinDL/issues/35)
* Added ReLU layer
* Added DepthwiseConv2D layer
* Added SeparableConv2D layer
* Added Reshape layer
* Added Cropping2D layer
* Added ZeroPadding2D layer
* Added NoGradients interface to indicate layers whose weights cannot be updated during training due to the lack of gradients in TensorFlow
* Added Model Zoo with the following models:
    * VGG'16
    * VGG'19
    * ResNet50
    * ResNet101
    * ResNet152
    * ResNet50V2
    * ResNet101V2
    * ResNet152V2
    * MobileNet
    * MobileNetV2
* Added ImageNet related preprocessing for each of the ModelZoo supported models: available in ModelZoo object and as a `sharpen` stage in the image preprocessing DSL
* Added model descriptions for models from ModelZoo (excluding MobileNet) designed with the Functional API in _org.jetbrains.kotlinx.dl.api.core.model_ package
* Added two implementations of the Dataset class: OnFlyImageDataset and OnHeapDataset
* Added topological sort for layers as nodes in the DAG model representation
* Added `shuffle` function support for both Dataset implementations
* Added the Kotlin-idiomatic DSL for image preprocessing with the following operations:
    * Load
    * Crop
    * Resize
    * Rotate
    * Rescale
    * Save
* Implemented label generation on the fly from the names of image folders
* Implemented `summary` method for the Functional API
* Added embedded datasets support (MNIST, FashionMNIST, Cifar'10, Cats & Dogs)

Bugs:
* Fixed a bug with BGR and RGB preprocessing in examples
* Fixed missed `useBias` field in convolutional layers

Internals improvements:
* Refactored: both Sequential and Functional models now inherit the GraphTrainableModel class
* Completed the Klaxon migration from 5.0.1 to 5.5
* Removed useless labels and data transformations before sending to  `Tensor.create(...)`

Infrastructure:
* Loaded the weights and JSON configurations of ModelZoo models to S3 storage
* Added a TeamCity build for the examples
* Loaded embedded datasets to S3 storage
* Removed dependencies from `jcenter`
* Moved an artifact to the Maven Central Repository
* Changed the groupId and artifactId
* Reduced the size of the downloaded `api` artifact from 65 MB to 650 KB by cleaning up resources and migrating the model and datasets to the S3 storage

Docs:
* Updated all the tutorials
* Updated the Readme.md

Examples:
* Renamed all the example's packages
* Regrouped examples between packages
* Added examples for training all ResNet models from scratch on the Cats & Dogs dataset
* Tuned hyper-parameters in all examples with VGG-like architecture to achieve convergence
* Added examples for the Image Preprocessing DSL
* Added examples for all available ModelZoo models, including additional training on the subset of the Cats & Dogs dataset
* Added ToyResNet examples (trained on the FashionMnist dataset)

Tests:
* Converted all examples to integration tests by refactoring `main` functions

# 0.1.1 (01/02/2021)
Features:
* Added [support batch processing for predictSoftly in #28](https://github.com/JetBrains/KotlinDL/issues/28)
* Converted [getXXX functions to properties in public API (layers and model classes)](https://github.com/JetBrains/KotlinDL/issues/29)
* Removed [a flag `verbose` from public API](https://github.com/JetBrains/KotlinDL/issues/20)
* Made logging based on a configuration.

Bugs:
* Fixed #25 [Suspiciously slow calls to `Sequential.predictSoftly`](https://github.com/JetBrains/KotlinDL/issues/25)
* Fixed #24 [reshapeFunction not initialized after model load](https://github.com/JetBrains/KotlinDL/issues/24)
* Fixed #22 [Exception in combination with Log4J](https://github.com/JetBrains/KotlinDL/issues/22)
* Added permission 'executable' making gradle wraper script runnable on Linux/Mac OS X systems

Internals improvements:
* Removed unnecessary copying of FloatArray to off-heap memory in `internalPredict` method

Docs:
* Added "Release check list" for release managers
* Updated Readme.md with new chapters about logging and fatJar issue
* Minor updates in "Tutorials"

# 0.1.0 (09/12/2020)
Features:
* Added @JvmStatic for companion methods

Examples:
* Provided support for VGG'16 and VGG'19 weights and models in examples
* Added links for loading all models and weights used in examples
* Moved direct file paths to property file
* Removed duplicated resources
* Transfer Learning examples are merged and improved
* Added description for all examples

Docs:
* Minor updates in "Transfer Learning Tutorial"

# 0.0.14 (20/11/2020)
Features:
* ReductionType support for loss function: SUM and SUM_OVER_BATCH_SIZE
* Added new loss functions: LogCosh, BinaryCrossEntropy, SquaredHinge

Tests:
* Added tests for all loss functions (in Eager Mode)
* Added tests for new scenarios of Keras weights loading

Docs:
* "Transfer Learning Tutorial" added
* Code of conduct and ChangeLog documents added
