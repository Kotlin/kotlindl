/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.dsl.functional
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.merge.*
import org.junit.jupiter.api.Test

class MergeLayersImportExportTest {
    @Test
    fun add() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Add("test_add")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun subtract() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Subtract("test_subtract")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun multiply() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Multiply("test_multiply")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun maximum() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Maximum("test_maximum")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun minimum() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Minimum("test_minimum")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun average() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Average("test_average")(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun concatenate() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input).apply { freeze() }
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input).apply { freeze() }
                    +Concatenate(name = "test_concatenate", axis = 1)(dense1, dense2)
                }
            }
        )
    }

    @Test
    fun dot() {
        LayerImportExportTest.run(
            functional {
                layers {
                    val input = +Input(10)
                    val dense1 = +Dense(name = "Dense1", outputSize = 5)(input)
                    val dense2 = +Dense(name = "Dense2", outputSize = 5)(input)
                    +Dot(name = "test_dot", axis = intArrayOf(1, 1), normalize = true)(dense1, dense2)
                }
            }
        )
    }
}