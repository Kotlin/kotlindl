/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PreprocessingFinalShapeTest {
    @Test
    fun resizeNoInputShape() {
        val preprocess = preprocess {
            transformImage {
                resize {
                    outputWidth = 100
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
            }
        }
        assertEquals(ImageShape(100, 100, null), preprocess.getFinalShape())
    }

    @Test
    fun resizeInputShape() {
        val preprocess = preprocess {
            transformImage {
                resize {
                    outputWidth = 100
                    outputHeight = 100
                    interpolation = InterpolationType.NEAREST
                }
            }
        }
        assertEquals(ImageShape(100, 100, 3), preprocess.getFinalShape(ImageShape(20, 20, 3)))
    }

    @Test
    fun cropImage() {
        val preprocess = preprocess {
            transformImage {
                crop {
                    left = 3
                    right = 11
                    top = 5
                    bottom = 7
                }
            }
        }
        assertEquals(ImageShape(186, 188, 3), preprocess.getFinalShape(ImageShape(200, 200, 3)))
    }

    @Test
    fun cropTwice() {
        val preprocess = preprocess {
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
        assertEquals(ImageShape(180, 180, 3), preprocess.getFinalShape(ImageShape(200, 200, 3)))
    }

    @Test
    fun resizeAndCrop() {
        val preprocess = preprocess {
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
        assertEquals(ImageShape(140, 90, 3), preprocess.getFinalShape(ImageShape(200, 200, 3)))
    }

    @Test
    fun rotateImage() {
        val preprocess = preprocess {
            transformImage {
                rotate {
                    degrees = 30f
                }
            }
        }
        assertEquals(ImageShape(200, 200, 3), preprocess.getFinalShape(ImageShape(200, 200, 3)))
    }

    @Test
    fun padImage() {
        val preprocess = preprocess {
            transformImage {
                pad {
                    top = 5
                    bottom = 7
                    left = 11
                    right = 13
                }
            }
        }
        assertEquals(ImageShape(324, 212, 1), preprocess.getFinalShape(ImageShape(300, 200, 1)))
    }

    @Test
    fun centerCropImage() {
        val preprocess = preprocess {
            transformImage {
                centerCrop { size = 15 }
            }
        }
        assertEquals(ImageShape(15, 15, 1), preprocess.getFinalShape(ImageShape(10, 20, 1)))
    }
}