/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.UpSampling3D
import org.junit.jupiter.api.Test

internal class UpSampling3DTest {
    private val input = arrayOf(
        arrayOf(
            arrayOf(
                arrayOf(
                    floatArrayOf(0.0f, 1.0f),
                    floatArrayOf(2.0f, 3.0f),
                    floatArrayOf(4.0f, 5.0f)
                ),
                arrayOf(
                    floatArrayOf(6.0f, 7.0f),
                    floatArrayOf(8.0f, 9.0f),
                    floatArrayOf(10.0f, 11.0f)
                )
            )
        )
    )

    @Test
    fun default() {
        val layer = UpSampling3D()
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f),
                        floatArrayOf(10.0f, 11.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f),
                        floatArrayOf(10.0f, 11.0f)
                    )
                ),
                arrayOf(
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f),
                        floatArrayOf(10.0f, 11.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f),
                        floatArrayOf(10.0f, 11.0f)
                    )
                )
            )
        )
        TODO("Implement test case")
    }

    @Test
    fun testWithUnequalSize() {
        val layer = UpSampling3D(size = intArrayOf(1, 2, 1))
        val expected = arrayOf(
            arrayOf(
                arrayOf(
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(0.0f, 1.0f),
                        floatArrayOf(2.0f, 3.0f),
                        floatArrayOf(4.0f, 5.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f)
                    ),
                    arrayOf(
                        floatArrayOf(6.0f, 7.0f),
                        floatArrayOf(8.0f, 9.0f),
                        floatArrayOf(10.0f, 11.0f)
                    )
                )
            )
        )
        TODO("Implement test case")
    }
}
