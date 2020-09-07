package api.tensor

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.op.Ops


class KTensor : AutoCloseable {
    private lateinit var session: Session

    lateinit var tensor: Tensor<Int>

    operator fun plus(kTensor: KTensor): KTensor {
        val otherTensor = kTensor.tensor
        require(tensor.shape()!!.contentEquals(otherTensor.shape())) {
            "The left tensor shape ${
                tensor.shape()!!
                    .contentToString()
            } must be equal the right tensor shape ${
                otherTensor.shape()!!
                    .contentToString()
            }."
        }

        var addTensor: Tensor<Int>

        Graph().use { g ->
            Session(g).use { session ->
                val tf = Ops.create(g)
                val aOps = tf.placeholder(Int::class.javaObjectType)
                val bOps = tf.placeholder(Int::class.javaObjectType)
                val addOps = tf.math.add(aOps, bOps)

                addTensor = session
                    .runner()
                    .feed(aOps.asOutput(), tensor)
                    .feed(bOps.asOutput(), otherTensor)
                    .fetch(addOps)
                    .run()[0] as Tensor<Int>

                val res = KTensor()
                res.tensor = addTensor

                return res
            }
        }
    }

    companion object {
        fun create(vector: IntArray): KTensor {
            val res = KTensor()
            res.tensor = Tensor.create(vector, Int::class.javaObjectType)

            return res
        }
    }

    override fun close() {
        session.close()
        tensor.close()
    }

    operator fun minus(kTensor: KTensor): KTensor {
        val otherTensor = kTensor.tensor
        require(tensor.shape()!!.contentEquals(otherTensor.shape())) {
            "The left tensor shape ${tensor.shape()!!
                .contentToString()} must be equal the right tensor shape ${otherTensor.shape()!!
                .contentToString()}."
        }

        var addTensor: Tensor<Int>

        Graph().use { g ->
            Session(g).use { session ->
                val tf = Ops.create(g)
                val aOps = tf.placeholder(Int::class.javaObjectType)
                val bOps = tf.placeholder(Int::class.javaObjectType)
                val addOps = tf.math.sub(aOps, bOps)

                addTensor = session
                    .runner()
                    .feed(aOps.asOutput(), tensor)
                    .feed(bOps.asOutput(), otherTensor)
                    .fetch(addOps)
                    .run()[0] as Tensor<Int>

                val res = KTensor()
                res.tensor = addTensor

                return res
            }
        }
    }

    operator fun times(kTensor: KTensor): KTensor {
        val otherTensor = kTensor.tensor
        require(tensor.shape()!!.contentEquals(otherTensor.shape())) {
            "The left tensor shape ${tensor.shape()!!
                .contentToString()} must be equal the right tensor shape ${otherTensor.shape()!!
                .contentToString()}."
        }

        var addTensor: Tensor<Int>

        Graph().use { g ->
            Session(g).use { session ->
                val tf = Ops.create(g)
                val aOps = tf.placeholder(Int::class.javaObjectType)
                val bOps = tf.placeholder(Int::class.javaObjectType)
                val addOps = tf.math.mul(aOps, bOps)

                addTensor = session
                    .runner()
                    .feed(aOps.asOutput(), tensor)
                    .feed(bOps.asOutput(), otherTensor)
                    .fetch(addOps)
                    .run()[0] as Tensor<Int>

                val res = KTensor()
                res.tensor = addTensor

                return res
            }
        }
    }
}
