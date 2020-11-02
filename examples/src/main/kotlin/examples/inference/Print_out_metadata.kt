/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference

import api.inference.savedmodel.SavedModel

private const val PATH_TO_MODEL = "api/src/main/resources/models/savedmodel"

fun main() {
    val model = SavedModel.load(PATH_TO_MODEL)

    println(model.toString())
}