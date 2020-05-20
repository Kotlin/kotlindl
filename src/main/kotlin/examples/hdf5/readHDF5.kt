package examples.hdf5


import io.jhdf.HdfFile
import io.jhdf.api.Group
import tf_api.keras.dataset.ImageDataset
import java.io.File

fun main() {
    val pathToLabels = "models/mnist/lenet_weights.hdf5"
    val realPathToLabels = ImageDataset::class.java.classLoader.getResource(pathToLabels).path.toString()

    val file = File(realPathToLabels)
    println(file.isFile)

    val hdfFile = HdfFile(file)

    hdfFile.use { hdfFile ->
        recursivePrintGroup(hdfFile, hdfFile, 0)
    }
}

fun recursivePrintGroup(hdfFile: HdfFile, group: Group, level: Int) {
    var level = level
    level++

    var indent = ""

    for (i in 1..level) {
        indent += "    "
    }

    for (node in group) {
        println(indent + node.name)


        for (entry in node.attributes.entries) {
            println(entry.value)
        }

        if (node is Group) {
            recursivePrintGroup(hdfFile, node, level)
        } else {
            println(node.path)
            val dataset = hdfFile.getDatasetByPath(node.path)
            val dims = arrayOf(dataset.dimensions)
            println(dims.contentDeepToString())

            when (dataset.dimensions.size) {
                4 -> {
                    val data = dataset.data as Array<Array<Array<FloatArray>>>
                    //println(data.contentDeepToString())
                }
                3 -> {
                    val data = dataset.data as Array<Array<FloatArray>>
                    //println(data.contentDeepToString())
                }
                2 -> {
                    val data = dataset.data as Array<FloatArray>
                    //println(data.contentDeepToString())
                }
                1 -> {
                    val data = dataset.data as FloatArray
                    //println(data.contentToString())
                }
            }
        }
    }
}





