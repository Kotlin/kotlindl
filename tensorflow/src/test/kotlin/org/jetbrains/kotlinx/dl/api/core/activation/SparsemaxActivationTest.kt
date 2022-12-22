package org.jetbrains.kotlinx.dl.api.core.activation

import org.junit.jupiter.api.Test

internal class SparsemaxActivationTest : ActivationTest() {
    private val inp1 = floatArrayOf(-5.0f, 1.0f, 2.0f)
    private val inp2 = floatArrayOf(-1.0f, 0.0f, 1.0f)
    private val inp3 = floatArrayOf(0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f)
    private val inp4 = floatArrayOf(-0.49786797f, -1.18600456f, -1.88243873f)
    private val inp3D = arrayOf(
        arrayOf(
            floatArrayOf(-0.49786797f, 1.32194696f, -2.99931375f),
            floatArrayOf(-1.18600456f, -2.11946466f, -2.44596843f),
            floatArrayOf(-1.88243873f, -0.92663564f, -0.61939515f)
        ),
        arrayOf(
            floatArrayOf(0.2329004f, -0.48483291f, 1.111317f),
            floatArrayOf(-1.7732865f, 2.26870462f, -2.83567444f),
            floatArrayOf(1.02280506f, -0.49617119f, 0.35213897f)
        ),
        arrayOf(
            floatArrayOf(-2.15767837f, -1.81139107f, 1.80446741f),
            floatArrayOf(2.80956945f, -1.11945493f, 1.15393569f),
            floatArrayOf(2.25833491f, 2.36763998f, -2.48973473f)
        )
    )

    @Test
    fun testSparsemaxOfOneDim() {
        val expected = floatArrayOf(0.0f, 0.0f, 1.0f)
        assertActivationFunction(SparsemaxActivation(), inp1, expected)

        val expected2 = floatArrayOf(0.0f, 0.0f, 1.0f)
        assertActivationFunction(SparsemaxActivation(), inp2, expected2)

        val expected3 = floatArrayOf(0.0f, 0.0f, 0.0f, 0.13333333f, 0.3333333f, 0.5333333f)
        assertActivationFunction(SparsemaxActivation(), inp3, expected3)

        val expected4 = floatArrayOf(0.8440683f, 0.1559317f, 0f)
        assertActivationFunction(SparsemaxActivation(), inp4, expected4)
    }

    @Test
    fun testSparsemaxOfMultiDim() {
        val inp = arrayOf(inp1, inp2)
        val exp = arrayOf(floatArrayOf(0.0f, 0.0f, 1.0f), floatArrayOf(0.0f, 0.0f, 1.0f))

        assertActivationFunction(SparsemaxActivation(), inp, exp)
    }

    @Test
    fun testSparsemaxOfMultiDimAxisIsZero() {
        // testing axis = 0
        val expWhenAxisIsZero = arrayOf(
            arrayOf(
                floatArrayOf(0.13461581f, 1f, 0f),
                floatArrayOf(0f, 0f, 0f),
                floatArrayOf(0f, 0f, 0.01423294f)
            ),
            arrayOf(
                floatArrayOf(0.86538419f, 0f, 0.1534248f),
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(0f, 0f, 0.98576706f)
            ),
            arrayOf(
                floatArrayOf(0f, 0f, 0.8465752f),
                floatArrayOf(1f, 0f, 1f),
                floatArrayOf(1f, 1f, 0f)
            )
        )

        assertActivationFunction(SparsemaxActivation(0), inp3D, expWhenAxisIsZero)
    }

    @Test
    fun testSparsemaxOfMultiDimAxisIsOne() {
        // testing axis = 1
        val expWhenAxisIsOne = arrayOf(
            arrayOf(
                floatArrayOf(0.8440683f, 1f, 0f),
                floatArrayOf(0.1559317f, 0f, 0f),
                floatArrayOf(0f, 0f, 1f)
            ),
            arrayOf(
                floatArrayOf(0.10504767f, 0f, 0.87958902f),
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(0.89495233f, 0f, 0.12041098f)
            ),
            arrayOf(
                floatArrayOf(0f, 0f, 0.82526586f),
                floatArrayOf(0.77561727f, 0f, 0.17473414f),
                floatArrayOf(0.22438273f, 1f, 0f)
            )
        )

        assertActivationFunction(SparsemaxActivation(1), inp3D, expWhenAxisIsOne)
    }

    @Test
    fun testSparsemaxOfMultiDimAxisIsTwo() {
        // Same as axis = -1 in 3D case
        val expWhenAxisIsTwo = arrayOf(
            arrayOf(
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(0.96673005f, 0.03326995f, 0f),
                floatArrayOf(0f, 0.34637976f, 0.65362024f)
            ),
            arrayOf(
                floatArrayOf(0.0607917f, 0f, 0.9392083f),
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(0.83533305f, 0f, 0.16466695f)
            ),
            arrayOf(
                floatArrayOf(0f, 0f, 1f),
                floatArrayOf(1f, 0f, 0f),
                floatArrayOf(0.44534747f, 0.55465253f, 0f)
            )
        )

        assertActivationFunction(SparsemaxActivation(2), inp3D, expWhenAxisIsTwo)
    }

    @Test
    fun testSparsemaxOfInf() {
        // TESTING NEGATIVE INF
        val inpNegative = arrayOf(
            floatArrayOf(0f, Float.NEGATIVE_INFINITY, 0f),
            floatArrayOf(0f, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY),
            floatArrayOf(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY)
        )
        val expNegative = arrayOf(
            floatArrayOf(0.5f, 0f, 0.5f),
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN)
        )

        assertActivationFunction(SparsemaxActivation(), inpNegative, expNegative)

        // TESTING POSITIVE INF
        val inpPositive = arrayOf(
            floatArrayOf(0f, Float.POSITIVE_INFINITY, 0f),
            floatArrayOf(0f, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY),
            floatArrayOf(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY)
        )

        val expPositive = arrayOf(
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN),
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN),
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN)
        )

        assertActivationFunction(SparsemaxActivation(), inpPositive, expPositive)

        //TESTING MIX
        val inpMix = arrayOf(
            floatArrayOf(0f, Float.NEGATIVE_INFINITY, 0f),
            floatArrayOf(0f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY),
            floatArrayOf(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
        )

        val expMix = arrayOf(
            floatArrayOf(0.5f, 0f, 0.5f),
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN),
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN)
        )

        assertActivationFunction(SparsemaxActivation(), inpMix, expMix)
    }

    @Test
    fun testSparsemaxOfAllZero() {
        // Proposition 2 part 1.1
        // sparsemax([0, 0, .. , 0]) = 1 / K where K is number of zeros in all zero vector
        val numberOfElements = 10
        val inp = FloatArray(numberOfElements) { 0f }
        val exp = FloatArray(numberOfElements) { 1f / numberOfElements }
        assertActivationFunction(SparsemaxActivation(), inp, exp)
    }

    @Test
    fun testSparsemaxOfToInf() {
        // Proposition 2 part 1.2
        // If you divide vector z with very small number
        // or multiply it with very large number
        // sparsemax(z * veryLargeNumber) is equal to one-hot like vector where one is the biggest number in vector z
        val veryLargeNumber = 9999f
        val inp =
            inp3D.map { arrayOfFloatArrays ->
                arrayOfFloatArrays.map { floats ->
                    floats.map { it * veryLargeNumber }.toFloatArray()
                }.toTypedArray()
            }.toTypedArray()
        val exp = arrayOf(
            arrayOf(
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(1f, 0f, 0f),
                floatArrayOf(0f, 0f, 1f)
            ),
            arrayOf(
                floatArrayOf(0f, 0f, 1f),
                floatArrayOf(0f, 1f, 0f),
                floatArrayOf(1f, 0f, 0f)
            ),
            arrayOf(
                floatArrayOf(0f, 0f, 1f),
                floatArrayOf(1f, 0f, 0f),
                floatArrayOf(0f, 1f, 0f)
            )
        )
        assertActivationFunction(SparsemaxActivation(), inp, exp)
    }

    @Test
    fun testSparsemaxOfAddConstant() {
        // Proposition 2 part 2. Adding constant to vector x doesn't change result of sparsemax
        // i.e. Sparsemax(x) = Sparsemax(x + c)

        val inp = floatArrayOf(1.1f, -0.2f, 0.5f, -0.5f, 1.7f, 1.2f, -5.7f, 1.8f)
        val c = (-100..100).random().toFloat()
        val exp = floatArrayOf(0f, 0f, 0f, 0f, 0.45f, 0f, 0f, 0.55f)
        assertActivationFunction(SparsemaxActivation(), inp, exp)
        assertActivationFunction(SparsemaxActivation(), inp.map { it + c }.toFloatArray(), exp)
    }

    @Test
    fun testSparsemaxOfPermutation() {
        // Proposition 2 part 3.
        // sparsemax(P x) = P sparsemax(x) where P is permutation matrix

        // (perm * inp) calculations are hand calculated these are just here for reference
        // val perm1 = arrayOf(floatArrayOf(0f, 1f, 0f), floatArrayOf(0f, 0f, 1f), floatArrayOf(1f, 0f, 0f))
        // val perm2 = arrayOf(floatArrayOf(1f, 0f, 0f), floatArrayOf(0f, 0f, 1f), floatArrayOf(0f, 1f, 0f))

        val inp =
            arrayOf(floatArrayOf(-5f, 1f, 2f), floatArrayOf(-1f, 0f, 1f), floatArrayOf(-1.2f, -5f, -1.3f))
        val exp =
            arrayOf(floatArrayOf(0f, 0f, 1f), floatArrayOf(0f, 0f, 1f), floatArrayOf(0.54999995f, 0f, 0.45000005f))
        assertActivationFunction(SparsemaxActivation(), inp, exp)

        // using perm1
        val inpPerm1 =
            arrayOf(floatArrayOf(-1f, 0f, 1f), floatArrayOf(-1.2f, -5f, -1.3f), floatArrayOf(-5f, 1f, 2f))
        val expPerm1 =
            arrayOf(floatArrayOf(0f, 0f, 1f), floatArrayOf(0.54999995f, 0f, 0.45000005f), floatArrayOf(0f, 0f, 1f))
        assertActivationFunction(SparsemaxActivation(), inpPerm1, expPerm1)

        // using perm2
        val inpPerm2 =
            arrayOf(floatArrayOf(-5f, 1f, 2f), floatArrayOf(-1.2f, -5f, -1.3f), floatArrayOf(-1f, 0f, 1f))
        val expPerm2 =
            arrayOf(floatArrayOf(0f, 0f, 1f), floatArrayOf(0.54999995f, 0f, 0.45000005f), floatArrayOf(0f, 0f, 1f))
        assertActivationFunction(SparsemaxActivation(), inpPerm2, expPerm2)
    }

    @Test
    fun testSparsemaxOfDifference() {
        // Proposition 2 part 4.
        // if z_i <= z_j then 0 <= sparsemax(Z)_j - sparsemax(Z)_i <= (z_j - z_i)

        // input is sorted
        val inp = floatArrayOf(-0.5f, -0.2f, 0f, 0.5f, 0.6f, 0.7f, 0.75f, 0.8f)
        val exp = floatArrayOf(0f, 0f, 0f, 0.03000003f, 0.13000005f, 0.23000002f, 0.28000003f, 0.33000004f)
        assertActivationFunction(SparsemaxActivation(), inp, exp) // we can say exp = sparsemax(inp).

        for (i in 1..inp.lastIndex) {
            val s = exp[i] - exp[i - 1]
            assert(s in 0f..(inp[i] - inp[i - 1]))
        }
    }
}
