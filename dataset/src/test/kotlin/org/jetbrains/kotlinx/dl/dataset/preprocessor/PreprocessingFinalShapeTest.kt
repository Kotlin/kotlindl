package org.jetbrains.kotlinx.dl.dataset.preprocessor

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
            }
            transformImage {
                resize {
                    outputWidth = 100
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
            }
        }
        assertEquals(ImageShape(100, 100, null), preprocess.finalShape)
    }

    @Test
    fun resizeInputShape() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(20, 20, 3)
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
            }
            transformImage {
                rotate {
                    degrees = 30f
                }
            }
        }
        assertEquals(ImageShape(200, 200, 3), preprocess.finalShape)
    }

    @Test
    fun padImage() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(300, 200, 1)
            }
            transformImage {
                pad {
                    top = 5
                    bottom = 7
                    left = 11
                    right = 13
                }
            }
        }
        assertEquals(ImageShape(324, 212, 1), preprocess.finalShape)
    }

    @Test
    fun centerCropImage() {
        val preprocess = preprocess {
            load {
                pathToData = File("test.jpg")
                imageShape = ImageShape(10, 20, 1)
            }
            transformImage {
                centerCrop { size = 15 }
            }
        }
        assertEquals(ImageShape(15, 15, 1), preprocess.finalShape)
    }
}