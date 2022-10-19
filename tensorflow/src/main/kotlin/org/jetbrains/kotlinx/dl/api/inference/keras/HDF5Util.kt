/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import io.jhdf.api.Group

/**
 * Helper function to print out file in hdf5 format for debugging purposes.
 */
public fun recursivePrintGroupInHDF5File(hdfFile: HdfFile, group: Group) {
    for (node in group) {
        println("[HDFUtil] Node: " + node.name)

        for ((key, value) in node.attributes) {
            println("[HDFUtil] attribute name: $key")
            if (value.isScalar) {
                println("[HDFUtil] attribute data: " + value.data.toString())
            } else if (value.data is Array<*>) {
                for (i in 0 until value.size.toInt())
                    println("[HDFUtil] attribute #$i data: " + (value.data as Array<*>)[i].toString())
            }
        }

        if (node is Group) {
            recursivePrintGroupInHDF5File(hdfFile, node)
        } else {
            println("[HDFUtil] Path to node: " + node.path)
            val dataset = hdfFile.getDatasetByPath(node.path)
            val dims = arrayOf(dataset.dimensions)
            println("[HDFUtil] Shape: " + dims.contentDeepToString())
        }
    }
}
