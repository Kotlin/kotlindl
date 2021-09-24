package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File
import java.net.URISyntaxException
import java.net.URL

/** Converts resource string path to the file. */
@Throws(URISyntaxException::class)
fun getFileFromResource(fileName: String): File {
    val classLoader: ClassLoader = object {}.javaClass.classLoader
    val resource: URL? = classLoader.getResource(fileName)
    return if (resource == null) {
        throw IllegalArgumentException("file not found! $fileName")
    } else {
        File(resource.toURI())
    }
}

private fun preprocessing(
    resizeTo: Pair<Int, Int>,
    i: Int
): Preprocessing {
    val preprocessing: Preprocessing = if (resizeTo.first == 224 && resizeTo.second == 224) {
        preprocess {
            transformImage {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
            }
        }
    } else {
        preprocess {
            transformImage {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.RGB
                }
                resize {
                    outputWidth = 299
                    outputHeight = 299
                }
            }
        }
    }
    return preprocessing
}