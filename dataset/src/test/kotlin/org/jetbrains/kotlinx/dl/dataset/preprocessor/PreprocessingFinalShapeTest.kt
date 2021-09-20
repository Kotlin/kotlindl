package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.io.File

class PreprocessingFinalShapeTest {
    @Test
    fun resizeNoInputShape() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                colorMode = ColorOrder.BGR
            }
            transformImage {
                resize {
                    outputWidth = 100
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
            }
        }
        assertEquals(ImageShape(100, 100, 3), preprocess.finalShape)
    }

    @Test
    fun resizeInputShape() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(20, 20, 3)
                colorMode = ColorOrder.BGR
            }
            transformImage {
                resize {
                    outputWidth = 100
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
            }
        }
        assertEquals(ImageShape(100, 100, 3), preprocess.finalShape)
    }

    @Test
    fun cropImage() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(200, 200, 3)
                colorMode = ColorOrder.BGR
            }
            transformImage {
                crop {
                    left = 3
                    right = 11
                    top = 5
                    bottom = 7
                }
            }
        }
        assertEquals(ImageShape(186, 188, 3), preprocess.finalShape)
    }

    @Test
    fun cropTwice() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(200, 200, 3)
                colorMode = ColorOrder.BGR
            }
            transformImage {
                crop {
                    left = 3
                    right = 11
                    top = 5
                    bottom = 7
                }
                crop {
                    left = 4
                    right = 2
                    top = 5
                    bottom = 3
                }
            }
        }
        assertEquals(ImageShape(180, 180, 3), preprocess.finalShape)
    }

    @Test
    fun resizeAndCrop() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(200, 200, 3)
                colorMode = ColorOrder.BGR
            }
            transformImage {
                resize {
                    outputWidth = 150
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
                crop {
                    left = 5
                    right = 5
                    top = 5
                    bottom = 5
                }
            }
        }
        assertEquals(ImageShape(140, 90, 3), preprocess.finalShape)
    }

    @Test
    fun rotateImage() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(200, 200, 3)
                colorMode = ColorOrder.BGR
            }
            transformImage {
                rotate {
                    degrees = 30f
                }
            }
        }
        assertEquals(ImageShape(200, 200, 3), preprocess.finalShape)
    }
}