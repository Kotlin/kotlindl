# 0.5.0 (14/12/2022) Inference on Android with ONNX Runtime
Features:
* Added Android inference support
  * Built Android artifacts for "impl", "onnx" and "visualization" modules [#422](https://github.com/Kotlin/kotlindl/issues/422)
  * Added Android-specific models to the model zoo
    * Classification [#438](https://github.com/Kotlin/kotlindl/issues/438):
      * `EfficientNet4Lite`
      * `MobilenetV1` 
    * Object Detection:
      * `SSDMobileNetV1` [#440](https://github.com/Kotlin/kotlindl/issues/440)
      * `EfficientDetLite0` [#443](https://github.com/Kotlin/kotlindl/issues/443)
    * Pose Detection [#442](https://github.com/Kotlin/kotlindl/issues/442):
      * `MoveNetSinglePoseLighting`
      * `MoveNetSinglePoseThunder`
    * Face Detection [#461](https://github.com/Kotlin/kotlindl/pull/461):
      * `UltraFace320`
      * `UltraFace640`
    * Face Alignment [#441](https://github.com/Kotlin/kotlindl/issues/441):
      * `Fan2d106` 
  * Implemented preprocessing operations working on Android `Bitmap` [#416](https://github.com/Kotlin/kotlindl/issues/416) 
    [#478](https://github.com/Kotlin/kotlindl/pull/478):
    * `Resize`
    * `Rotate`
    * `Crop`
    * `ConvertToFloatArray`
  * Added utility functions to convert `ImageProxy` to `Bitmap` [#458](https://github.com/Kotlin/kotlindl/pull/458)
  * Added `NNAPI` execution provider [#420](https://github.com/Kotlin/kotlindl/issues/420)
  * Added api to create `OnnxInferenceModel` from the `ByteArray` representation [#415](https://github.com/Kotlin/kotlindl/issues/415)
  * Introduced a gradle task to download model hub models before the build [#444](https://github.com/Kotlin/kotlindl/issues/444)
  * Added utility functions to draw detection results on Android `Canvas` [#450](https://github.com/Kotlin/kotlindl/pull/450)
* Implemented new preprocessing API [#425](https://github.com/Kotlin/kotlindl/pull/425)
  * Introduced an `Operation` interface to represent a preprocessing operation for any input and output
  * Added `PreprocessingPipeline` class to combine operations together in a type-safe manner
  * Re-implemented old operations with the new API
  * Added convenience functions such as `pipeline` to start a new preprocessing pipeline, 
    `call` to invoke operations defined elsewhere, `onResult` to access intermediate preprocessing results
  * Converted `ModelType#preprocessInput` function to `Operation` [#429](https://github.com/Kotlin/kotlindl/pull/429)
  * Converted common preprocessing functions for models trained on ImageNet to `Operation` [#429](https://github.com/Kotlin/kotlindl/pull/429)
* Added new ONNX features
  * Added execution providers support (`CPU`, `CUDA`, `NNAPI`) and convenient extensions for inference with them 
    [#386](https://github.com/Kotlin/kotlindl/issues/386)
  * Introduced `OnnxInferenceModel#predictRaw` function which allows custom `OrtSession.Result` processing and extension functions
    to extract common data types from the result [#465](https://github.com/Kotlin/kotlindl/pull/465)
  * Added validation of input shape [#385](https://github.com/Kotlin/kotlindl/issues/385)
* Added `Imagenet` enum to represent different Imagenet dataset labels and added support for zero indexed COCO labels
  [#438](https://github.com/Kotlin/kotlindl/issues/438) [#446](https://github.com/Kotlin/kotlindl/pull/446)
* Implemented unified summary printing for Tensorflow and ONNX models [#368](https://github.com/Kotlin/kotlindl/issues/368)
* Added `FlatShape` interface to allow manipulating the detected shapes in a unified way [#480](https://github.com/Kotlin/kotlindl/pull/480)
* Introduced `DataLoader` interface for loading and preprocessing data for dataset implementations [#424](https://github.com/Kotlin/kotlindl/pull/424)
* Improved swing visualization utilities [#379](https://github.com/Kotlin/kotlindl/issues/379)
  [#388](https://github.com/Kotlin/kotlindl/issues/388)
* Simplified `Layer` interface to leave only `build` function to be implemented and remove explicit output shape computation
  [#408](https://github.com/Kotlin/kotlindl/pull/408)  

Breaking changes:
* Refactored module structure and packages [#412](https://github.com/Kotlin/kotlindl/pull/412) [#469](https://github.com/Kotlin/kotlindl/pull/469)
  * Extracted "tensorflow" module for learning and inference with Tensorflow backend
  * Extracted "impl" module for implementation classes and utilities
  * Moved preprocessing operation implementations to the "impl" module
  * Removed dependency of "api" module on the "dataset" module
  * Changed packages for "api", "impl", "dataset" and "onnx" so that they match the corresponding module name
* Preprocessing classes such as `Preprocessing`, `ImagePreprocessing`, `ImagePreprocessor`,
  `ImageSaver`, `ImageShape`, `TensorPreprocessing`, `Preprocessor` got removed in favor of the new preprocessing API [#425](https://github.com/Kotlin/kotlindl/pull/425)
* Removed `Sharpen` preprocessor since the `ModelType#preprocessor` field was introduced, which can be used in the preprocessing
  pipeline using the `call` function [#429](https://github.com/Kotlin/kotlindl/pull/429)

Bugfixes:
* Fix loading of jpeg files not supported by standard java ImageIO [#384](https://github.com/Kotlin/kotlindl/issues/384)
* Updated ONNX Runtime version to enable inference on M1 chips [#361](https://github.com/Kotlin/kotlindl/issues/361)
* Fixed channel ordering in for image recognition models [#400](https://github.com/Kotlin/kotlindl/issues/400)
* Avoid warnings from `loadWeightsForFrozenLayers` function for layers without parameters [#382](https://github.com/Kotlin/kotlindl/issues/382)

New documentation and examples:
* [Inference with KotlinDL and ONNX Runtime on desktop and Android](https://github.com/Kotlin/kotlindl/blob/master/docs/inference_onnx_model.md)
* [KotlinDL ONNX Model Zoo](https://github.com/Kotlin/kotlindl/blob/master/docs/onnx_model_zoo.md)
* [Sample Android App](https://github.com/Kotlin/kotlindl-app-sample)

# 0.4.0 (01/06/2022) Pose Detection, EfficientDet for Object Detection and EfficientNet for Image Recognition
Features:
* Added the PoseNet model family to the ONNX Model Hub. [#269](https://github.com/Kotlin/kotlindl/issues/269)
  * MoveNetSinglePoseLighting
  * MoveNetMultiPoseLighting
  * MoveNetSinglePoseThunder
* Added the EfficientDet model family to the ONNX Model Hub. [#304](https://github.com/Kotlin/kotlindl/issues/304)
  * EfficientDetD0
  * EfficientDetD1
  * EfficientDetD2
  * EfficientDetD3
  * EfficientDetD4
  * EfficientDetD5 
  * EfficientDetD6 
* Added SSD-Mobilenet-v1 model to the ONNX Model Hub. [#296](https://github.com/Kotlin/kotlindl/issues/296)
* Added EfficientNet model family to the ONNX Model Hub. [#264](https://github.com/Kotlin/kotlindl/issues/264)
  * EfficientNetB0 ( + noTop model)
  * EfficientNetB1 ( + noTop model)
  * EfficientNetB2 ( + noTop model)
  * EfficientNetB3 ( + noTop model)
  * EfficientNetB4 ( + noTop model)
  * EfficientNetB5 ( + noTop model)
  * EfficientNetB6 ( + noTop model)
  * EfficientNetB7 ( + noTop model)
* Added NoTop models to the TensorFlow Model Hub. [#281](https://github.com/Kotlin/kotlindl/issues/281)
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
  * NasNetMobile
  * NasNetLarge
  * DenseNet121
  * DenseNet169
  * DenseNet201
  * Xception
  * Inception
* Added new `Dot` layer and `Conv1DTranspose`, `Conv2DTranspose`, `Conv3DTranspose` layers.
  [#144](https://github.com/Kotlin/kotlindl/issues/144) [#124](https://github.com/Kotlin/kotlindl/issues/124)
* Added new activation functions: `SparsemaxActivation` and `SoftShrinkActivation`.
  [#171](https://github.com/Kotlin/kotlindl/issues/171) [#170](https://github.com/Kotlin/kotlindl/issues/170)
* Added new `Padding`, `CenterCrop`, `Convert`, `Grayscale` image preprocessors and `Normalizing` tensor preprocessor.
  [#203](https://github.com/Kotlin/kotlindl/issues/203) [#201](https://github.com/Kotlin/kotlindl/issues/201)
  [#202](https://github.com/Kotlin/kotlindl/issues/202) [#204](https://github.com/Kotlin/kotlindl/issues/204)

  
Examples and tutorials:
* Added an example of Image preprocessing DSL usage with TensorFlow model. [#292](https://github.com/Kotlin/kotlindl/issues/292).
* Added examples for Object Detection with [EfficientDet2](https://github.com/Kotlin/kotlindl/tree/master/examples/src/main/kotlin/examples/onnx/objectdetection/efficientdet).
* Added examples for Object Detection with [SSD-Mobilenet-v1](https://github.com/Kotlin/kotlindl/tree/master/examples/src/main/kotlin/examples/onnx/objectdetection/ssdmobile).
* Added examples for Pose Detection with [different models](https://github.com/Kotlin/kotlindl/tree/master/examples/src/main/kotlin/examples/onnx/posedetection).
* Added examples for Image Recognition with different models from [EfficientNet model family](https://github.com/Kotlin/kotlindl/tree/master/examples/src/main/kotlin/examples/onnx/cv/efficicentnet).
* Added examples for fine-tuning of [noTop ResNet model](https://github.com/Kotlin/kotlindl/blob/master/examples/src/main/kotlin/examples/transferlearning/modelhub/resnet/Example_5_ResNet50_prediction_additional_training_noTop.kt).
* Added an example for new Image Preprocessing DSL operator [Normalize](https://github.com/Kotlin/kotlindl/blob/master/examples/src/main/kotlin/examples/dataset/NormalizeExample.kt).
* Added an example for Linear Regression model training with [two metrics](https://github.com/Kotlin/kotlindl/blob/master/examples/src/main/kotlin/examples/ml/LinearRegressionWithTwoMetrics.kt).
* Added an example for new [Functional API DSL](https://github.com/Kotlin/kotlindl/blob/master/examples/src/main/kotlin/examples/transferlearning/toyresnet/Example_4_ToyResNet_describe_and_train_DSL.kt) with ToyResNet model. 
* Added an example for training LeNet model with [multiple callbacks support](https://github.com/Kotlin/kotlindl/blob/master/examples/src/main/kotlin/examples/cnn/mnist/advanced/LeNetWithMultipleCallbacks.kt).

API changes:
* Introduced new DSL for creating `Sequential` and `Functional` models. [#133](https://github.com/Kotlin/kotlindl/issues/133)
* Added support for multiple `Callbacks` in `fit()`, `evaluate()`, `predict()` instead of `compile()`.
  [#270](https://github.com/Kotlin/kotlindl/issues/270)
* Added support for multiple metrics. [#298](https://github.com/Kotlin/kotlindl/issues/298)
* Added support for the model reset. [#271](https://github.com/Kotlin/kotlindl/issues/271)
* Replaced `Long` parameters with `Integer` ones in convolutional, average pool and max pool layers.
  [#273](https://github.com/Kotlin/kotlindl/issues/273)
* Moved loading section out of image preprocessing. [#322](https://github.com/Kotlin/kotlindl/issues/322)
* Remove obsolete `CustomPreprocessor` interface. [#257](https://github.com/Kotlin/kotlindl/pull/257)
* Supported exporting tensor data to `BufferedImage` [#293](https://github.com/Kotlin/kotlindl/issues/293)

Internal API changes:
* Introduced new abstraction for layer parameters -- `KVariable`. [#324](https://github.com/Kotlin/kotlindl/pull/324)
* Moved some of the `Layer` functionality to the new interfaces `ParametrizedLayer` and `TrainableLayer`.
  [#217](https://github.com/Kotlin/kotlindl/issues/217)

Bug fixes:
* Added support for correct loading of `isTrainable` status from Keras. [#153](https://github.com/Kotlin/kotlindl/issues/153)
* Generalized `Reshape` layer to higher dimensions. [#249](https://github.com/Kotlin/kotlindl/pull/249)
* Fixed incorrect bounding box coordinates in ObjectDetection and FaceDetection examples. [#279](https://github.com/Kotlin/kotlindl/issues/279)
* Fixed `toString` methods for layer classes. [#301](https://github.com/Kotlin/kotlindl/issues/301)
* Set all the optimizers to have `useLocking = True` [#305](https://github.com/Kotlin/kotlindl/issues/305)
* Fixed a bug with silently skipped layers in topological sort. [#314](https://github.com/Kotlin/kotlindl/issues/314)
* Fixed `loadModelLayersFromConfiguration` recursively calling itself. [#319](https://github.com/Kotlin/kotlindl/pull/319)
* Fixed `GraphTrainableModel#internalPredict` for multi-dimensional predictions. [#327](https://github.com/Kotlin/kotlindl/issues/327)
* Fixed `Orthogonal` initializer. [#348](https://github.com/Kotlin/kotlindl/pull/348)
* Fixed initialization of the variables in `GraphTrainableModel`. [#355](https://github.com/Kotlin/kotlindl/pull/355)
* Add model output type checks for `OnnxInferenceModel`. [#356](https://github.com/Kotlin/kotlindl/pull/356)
* Fixed `IndexOutOfBoundsException` in the `Dot` layer. [#357](https://github.com/Kotlin/kotlindl/pull/357)
* Fixed import and export issues:
  - Fixed layers import and export and added tests. [#329](https://github.com/Kotlin/kotlindl/issues/329)
    [#341](https://github.com/Kotlin/kotlindl/issues/341) [#360](https://github.com/Kotlin/kotlindl/pull/360)
  - Fixed exporting `Ones`, `Zeros`, `Orthogonal` and `ParametrizedTruncatedNormal` initializers.
    [#331](https://github.com/Kotlin/kotlindl/issues/331)
* Updated log4j version.
* Fixed the group of examples with ToyResNet.


# 0.3.0 (28/09/2021) ONNX for inference and transfer learning and ONNX Model Hub
Features:
* Implemented the [copying for the Functional and Sequential models](https://github.com/JetBrains/KotlinDL/issues/40)
* Implemented the [copying for the TensorFlow-based Inference Model](https://github.com/JetBrains/KotlinDL/issues/178)
* Implemented the [experimental ONNX integration](https://github.com/JetBrains/KotlinDL/issues/184):
   * added new 'onnx' module
   * added the ONNXModel implementing the common InferenceModel interface
   * ONNX model could be used as a preprocessing stage for the TensorFlow model
   * prepared ONNX model without top layers could be fine-tuned via training of top layers implemented with TensorFlow-based layers
* Added SSD and YOLOv4 object detection models to the Model Hub
* Added Fan2D106 face alignment model to the Model Hub
* Added SSDObjectDetectionModel with the easy API for object detection, including pre- and post-processing
* Added a few models in ONNX format to the Model Hub
   * ResNet18
   * ResNet34
   * ResNet50
   * ResNet101 
   * ResNet152 
   * ResNet18V2
   * ResNet34V2
   * ResNet50V2 
   * ResNet101V2 
   * ResNet152V2
   * EfficientNetV4
* Added [new TensorFlow-based models to the Model Zoo (or Model Hub)](https://github.com/JetBrains/KotlinDL/issues/101): 
   * NasNetMobile
   * NasNetLarge
   * DenseNet121
   * DenseNet169
   * DenseNet201
   * Xception 
* Added [ResNet18 and ResNet34 TensorFlow-based models to ModelZoo](https://github.com/JetBrains/KotlinDL/issues/175)
* Added [L1 and L2 regularization to the layers](https://github.com/JetBrains/KotlinDL/issues/83)
* Added [Identity initializer](https://github.com/JetBrains/KotlinDL/issues/50)
* Added [Orthogonal initializer](https://github.com/JetBrains/KotlinDL/issues/51)
* Added [Softmax activation layer](https://github.com/JetBrains/KotlinDL/issues/52)
* Added [LeakyReLU activation layer](https://github.com/JetBrains/KotlinDL/issues/53)
* Added [PReLU activation layer](https://github.com/JetBrains/KotlinDL/issues/54)
* Added [ELU activation layer](https://github.com/JetBrains/KotlinDL/issues/55)
* Added [ThresholdedReLU activation layer](https://github.com/JetBrains/KotlinDL/issues/56)
* Added [Conv1D layer](https://github.com/JetBrains/KotlinDL/issues/59)
* Added [MaxPooling1D layer](https://github.com/JetBrains/KotlinDL/issues/60)
* Added [AveragePooling1D layer](https://github.com/JetBrains/KotlinDL/issues/61)
* Added [GlobalMaxPooling1D layer](https://github.com/JetBrains/KotlinDL/issues/62)
* Added [GlobalAveragePooling1D layer](https://github.com/JetBrains/KotlinDL/issues/63)
* Added [Conv3D layer](https://github.com/JetBrains/KotlinDL/issues/79)
* Added [MaxPooling3D layer](https://github.com/JetBrains/KotlinDL/issues/80)
* Added [AveragePooling3D layer](https://github.com/JetBrains/KotlinDL/issues/81)
* Added [GlobalAveragePooling3D layer](https://github.com/JetBrains/KotlinDL/issues/82)
* Added [GlobalMaxPool2D layer](https://github.com/JetBrains/KotlinDL/issues/116)
* Added [GlobalMaxPool3D layer](https://github.com/JetBrains/KotlinDL/issues/117)
* Added [Cropping1D and Cropping3D layers](https://github.com/JetBrains/KotlinDL/issues/121)
* Added [Permute layer](https://github.com/JetBrains/KotlinDL/issues/142)
* Added [RepeatVector layer](https://github.com/JetBrains/KotlinDL/issues/123)
* Added [UpSampling1D, UpSampling2D and UpSampling3D layers](https://github.com/JetBrains/KotlinDL/issues/143)
* Added [Gelu activation function](https://github.com/JetBrains/KotlinDL/issues/165)
* Added [HardShrink activation function](https://github.com/JetBrains/KotlinDL/issues/166)
* Added [LiSHT activation function](https://github.com/JetBrains/KotlinDL/issues/167)
* Added [Mish activation function](https://github.com/JetBrains/KotlinDL/issues/168)
* Added [Snake activation function](https://github.com/JetBrains/KotlinDL/issues/169)
* Added [Tanh shrink activation function](https://github.com/JetBrains/KotlinDL/issues/172)
* Added [TimeStopping callback](https://github.com/JetBrains/KotlinDL/issues/174)

Bugs:
* Added [missed loaders for the ReLU and ELU activation layers](https://github.com/JetBrains/KotlinDL/issues/78)
* Add [model export for a few layers (Concatenate, DepthwiseConv2D, SeparableConv2D) missed in ModelSaver.kt](https://github.com/JetBrains/KotlinDL/issues/87)
* Fixed the use-case when [ModelSaver fails on saving Input with 2d and 3d tensors](https://github.com/JetBrains/KotlinDL/issues/160)
* Fixed a [StackOverflowError in objectDetectionSSD.kt example](https://github.com/JetBrains/KotlinDL/issues/230)
* Fixed a problem with the [confusing logs during weights loading from .h5 file](https://github.com/JetBrains/KotlinDL/issues/155)
* Fixed the Windows separator usage instead of [File.separator in the Save and Load preprocessors](https://github.com/JetBrains/KotlinDL/issues/226)
* Fixed the [incorrect temporary folder for the cat-vs-dogs dataset](https://github.com/JetBrains/KotlinDL/issues/161)
* Fixed the problem when [ImageConverter and Loading do not close opened streams](https://github.com/JetBrains/KotlinDL/issues/228)
* Fixed the [Image Preprocessing DSL issues](https://github.com/JetBrains/KotlinDL/issues/208)
* Reduced time complexity of FloatArray::argmax to linear

API breaking changes:
* Renamed ModelZoo to the ModelHub
* Changed the ImagePreprocessing DSL: loading and saving are moved to the separate level of DSL
* Changed the [TrainableModel::summary API to return ModelSummary](https://github.com/JetBrains/KotlinDL/issues/135)

Infrastructure:
* Loaded the weights and JSON configurations of the newly added ModelHub models to S3 storage
* [Moved ImageDSL and Dataset API to the separate 'dataset' module](https://github.com/JetBrains/KotlinDL/issues/180)
* Added a new 'visualization' module with the basic support for painting on Swing and in Jupyter Notebook with lets-plot
* Transformed the project from the single-module project to the multi-module project

Docs:
* Created [website with API Documentation from KDoc via Dokka](https://github.com/JetBrains/KotlinDL/issues/71)
* Added support for the multiple version API Documentation from KDoc via Dokka 
* Updated all existing tutorials
* Updated the Readme.md
* Updated the existing KDocs
* Added a new tutorial about ONNX models usage
* Added a new tutorial about Transfer Learning with ONNX ResNet no-top model and TensorFlow

Examples:
* Added an [example](https://github.com/JetBrains/KotlinDL/blob/master/examples/src/main/kotlin/examples/onnx/objectdetection/ssd/objectDetectionSSD.kt) of SSDObjectDetectionModel usage and visualisation of the detected objects on the Swing panel
* Added an [example](https://github.com/JetBrains/KotlinDL/blob/master/examples/src/main/kotlin/examples/onnx/faces/predictionFan2D106.kt) of Fan2D106 (face alignment) model and landmarks visualisation on the Swing panel
* Added an [example](https://github.com/JetBrains/KotlinDL/blob/master/examples/src/main/kotlin/examples/onnx/cv/custom/additionalTrainingWithTensorFlow.kt) where the prepared ONNX model without top layers is fine-tuned via training of top layers implemented with TensorFlow-based layers
* Added a lot of examples for the newly added to the ModelHub models (ONNX-based and TensorFlow-based)
* Added an [example](https://github.com/JetBrains/KotlinDL/blob/master/examples/src/main/kotlin/examples/visualization/SoundNetFSDDVisualization.kt) with the model SoundNet trained on Free Spoken Digits Dataset to classify the audio 
* Updated ['visualization'](https://github.com/JetBrains/KotlinDL/tree/master/examples/src/main/kotlin/examples/visualization) examples with the new Batik and lets-plot support

Tests:
* Added tests for ModelLoading
* Added tests for InputLayer
* Added tests for all newly added layers

# 0.2.0 (17/05/2021) Functional API, Model Zoo and Image Preprocessing DSL
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
    * Sharpen
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
