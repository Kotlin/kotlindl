package examples.tensor

import api.tensor.KTensorInt32


fun main() {
    // Rank 1 Tensor
    val vector1 = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val vector2 = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    val t1 = KTensorInt32.create(vector1)
    val t2 = KTensorInt32.create(vector2)

    val addTensor = t1 + t2

    println("DataType: " + addTensor.tensor.dataType().name)
    println("NumElements: " + addTensor.tensor.numElements())
    println("NumDimensions: " + addTensor.tensor.numDimensions())

    addTensor.tensor.use { t ->
        val copy = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        println("Copied element: " + t.copyTo(copy)[9])
    }

    val subTensor = t1 - t2

    subTensor.tensor.use { t ->
        val copy = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        println("Copied element: " + t.copyTo(copy)[9])
    }

    val mulTensor = t1 * t2

    mulTensor.tensor.use { t ->
        val copy = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        println("Copied element: " + t.copyTo(copy)[9])
    }
}