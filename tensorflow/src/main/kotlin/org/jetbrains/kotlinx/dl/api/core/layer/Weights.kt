/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.inference.toMultiDimensionalArray
import org.jetbrains.kotlinx.dl.api.inference.toTensorList


/**
 *  Weights of this layer. Require parent model to be set on the layer.
 */
public var Layer.weights: Map<String, Array<*>>
    get() {
        if (this !is ParametrizedLayer) return emptyMap()

        val model = parentModel
        requireNotNull(model) { "Layer '$name' is not related to any model" }

        val runner = model.session.runner()
        variables.map { it.variable }.forEach(runner::fetch)
        val weights = runner.run().toTensorList().map { it.toMultiDimensionalArray() }

        return variables.map { it.name }.zip(weights).toMap()
    }
    set(weights) {
        if (this !is ParametrizedLayer) return

        val model = parentModel
        requireNotNull(model) { "Layer '$name' is not related to any model" }

        for (variable in variables) {
            val value = weights[variable.name]
            if (value != null) {
                model.fill(variable, value)
            }
        }
    }