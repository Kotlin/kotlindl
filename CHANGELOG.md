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

Internals improvements:
* Removed unnecessary copying of FloatArray to off-heap memory in `internalPredict` method

Docs:
* Added "Release check list" for release managers
* Updated Readme.md with new chapters about logging and fatJar issue

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
