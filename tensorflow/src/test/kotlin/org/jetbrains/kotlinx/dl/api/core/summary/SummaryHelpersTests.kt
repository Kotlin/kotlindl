package org.jetbrains.kotlinx.dl.api.core.summary

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.summary.print
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

internal class SummaryHelpersTests {
    private val sequentialModel = TfModelSummary(
        type = "Sequential",
        name = "My sequential NN model",
        layersSummaries = listOf(
            LayerSummary("input_1", "Input", TensorShape(-1, 28, 28, 1), 0, emptyList()),
            LayerSummary("conv2d_1", "Conv2D", TensorShape(-1, 28, 28, 32), 832, emptyList()),
            LayerSummary("maxPool_1", "MaxPool2D", TensorShape(-1, 14, 14, 32), 0, emptyList()),
            LayerSummary("conv2d_2", "Conv2D", TensorShape(-1, 14, 14, 64), 51264, emptyList()),
            LayerSummary("maxPool_2", "MaxPool2D", TensorShape(-1, 7, 7, 64), 0, emptyList()),
            LayerSummary("flatten_1", "Flatten", TensorShape(-1, 3136), 0, emptyList()),
            LayerSummary("dense_1", "Dense", TensorShape(-1, 512), 1606144, emptyList()),
            LayerSummary("dense_2", "Dense", TensorShape(-1, 10), 5130, emptyList())
        ),
        trainableParamsCount = 1663370,
        frozenParamsCount = 0
    )

    private val functionalModel = TfModelSummary(
        type = "Functional",
        name = "My functional NN model",
        layersSummaries = listOf(
            LayerSummary("input_1", "Input", TensorShape(-1, 28, 28, 1), 0, emptyList()),
            LayerSummary("conv2D_1", "Conv2D", TensorShape(-1, 26, 26, 32), 320, listOf("input_1")),
            LayerSummary("conv2D_2", "Conv2D", TensorShape(-1, 24, 24, 64), 18496, listOf("conv2D_1")),
            LayerSummary("maxPool2D", "MaxPool2D", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_2")),
            LayerSummary("conv2D_4", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("maxPool2D")),
            LayerSummary("conv2D_5", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("conv2D_4")),
            LayerSummary("add", "Add", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_5", "maxPool2D")),
            LayerSummary("conv2D_6", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("add")),
            LayerSummary("conv2D_7", "Conv2D", TensorShape(-1, 8, 8, 64), 36928, listOf("conv2D_6")),
            LayerSummary("add_1", "Add", TensorShape(-1, 8, 8, 64), 0, listOf("conv2D_7", "add")),
            LayerSummary("conv2D_8", "Conv2D", TensorShape(-1, 6, 6, 64), 36928, listOf("add_1")),
            LayerSummary("globalAvgPool2D", "GlobalAvgPool2D", TensorShape(-1, 64), 0, listOf("conv2D_8")),
            LayerSummary("dense_1", "Dense", TensorShape(-1, 256), 16640, listOf("globalAvgPool2D")),
            LayerSummary("dense_2", "Dense", TensorShape(-1, 10), 2570, listOf("dense_1"))
        ),
        trainableParamsCount = 2570,
        frozenParamsCount = 220096
    )

    @Test
    fun formatSequentialModelSummary() {
        sequentialModel.print()
        assertEquals(
            listOf(
                "==============================================================================",
                "Model type: Sequential",
                "Model name: My sequential NN model",
                "______________________________________________________________________________",
                "Layer (type)                           Output Shape              Param #      ",
                "==============================================================================",
                "input_1(Input)                         [None, 28, 28, 1]         0            ",
                "______________________________________________________________________________",
                "conv2d_1(Conv2D)                       [None, 28, 28, 32]        832          ",
                "______________________________________________________________________________",
                "maxPool_1(MaxPool2D)                   [None, 14, 14, 32]        0            ",
                "______________________________________________________________________________",
                "conv2d_2(Conv2D)                       [None, 14, 14, 64]        51264        ",
                "______________________________________________________________________________",
                "maxPool_2(MaxPool2D)                   [None, 7, 7, 64]          0            ",
                "______________________________________________________________________________",
                "flatten_1(Flatten)                     [None, 3136]              0            ",
                "______________________________________________________________________________",
                "dense_1(Dense)                         [None, 512]               1606144      ",
                "______________________________________________________________________________",
                "dense_2(Dense)                         [None, 10]                5130         ",
                "______________________________________________________________________________",
                "==============================================================================",
                "Total trainable params: 1663370",
                "Total frozen params: 0",
                "Total params: 1663370",
                "______________________________________________________________________________"
            ),
            sequentialModel.format()
        )
    }

    @Test
    fun formatFunctionalModelSummary() {
        assertEquals(
            listOf(
                "==========================================================================================================",
                "Model type: Functional",
                "Model name: My functional NN model",
                "__________________________________________________________________________________________________________",
                "Layer (type)                           Output Shape              Param #       Connected to               ",
                "==========================================================================================================",
                "input_1(Input)                         [None, 28, 28, 1]         0                                        ",
                "__________________________________________________________________________________________________________",
                "conv2D_1(Conv2D)                       [None, 26, 26, 32]        320           input_1                    ",
                "__________________________________________________________________________________________________________",
                "conv2D_2(Conv2D)                       [None, 24, 24, 64]        18496         conv2D_1                   ",
                "__________________________________________________________________________________________________________",
                "maxPool2D(MaxPool2D)                   [None, 8, 8, 64]          0             conv2D_2                   ",
                "__________________________________________________________________________________________________________",
                "conv2D_4(Conv2D)                       [None, 8, 8, 64]          36928         maxPool2D                  ",
                "__________________________________________________________________________________________________________",
                "conv2D_5(Conv2D)                       [None, 8, 8, 64]          36928         conv2D_4                   ",
                "__________________________________________________________________________________________________________",
                "add(Add)                               [None, 8, 8, 64]          0             conv2D_5                   ",
                "                                                                               maxPool2D                  ",
                "__________________________________________________________________________________________________________",
                "conv2D_6(Conv2D)                       [None, 8, 8, 64]          36928         add                        ",
                "__________________________________________________________________________________________________________",
                "conv2D_7(Conv2D)                       [None, 8, 8, 64]          36928         conv2D_6                   ",
                "__________________________________________________________________________________________________________",
                "add_1(Add)                             [None, 8, 8, 64]          0             conv2D_7                   ",
                "                                                                               add                        ",
                "__________________________________________________________________________________________________________",
                "conv2D_8(Conv2D)                       [None, 6, 6, 64]          36928         add_1                      ",
                "__________________________________________________________________________________________________________",
                "globalAvgPool2D(GlobalAvgPool2D)       [None, 64]                0             conv2D_8                   ",
                "__________________________________________________________________________________________________________",
                "dense_1(Dense)                         [None, 256]               16640         globalAvgPool2D            ",
                "__________________________________________________________________________________________________________",
                "dense_2(Dense)                         [None, 10]                2570          dense_1                    ",
                "__________________________________________________________________________________________________________",
                "==========================================================================================================",
                "Total trainable params: 2570",
                "Total frozen params: 220096",
                "Total params: 222666",
                "__________________________________________________________________________________________________________"
            ),
            functionalModel.format()
        )
    }

    @Test
    fun formatFunctionalModelSummaryWithCustomFormatting() {
        functionalModel.customFormat(
            layerNameColumnName = "Name of the layer along with its type",
            outputShapeColumnName = "Output shape",
            paramsCountColumnName = "# of parameters",
            connectedToColumnName = "Inbound connections",
            columnSeparator = "::",
            lineSeparatorSymbol = '.',
            thickLineSeparatorSymbol = ':',
        ).forEach { println(it) }

        assertEquals(
            listOf(
                ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",
                "Model type: Functional",
                "Model name: My functional NN model",
                "...............................................................................................",
                "Name of the layer along with its type::Output shape      ::# of parameters::Inbound connections",
                ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",
                "input_1(Input)                       ::[None, 28, 28, 1] ::0              ::                   ",
                "...............................................................................................",
                "conv2D_1(Conv2D)                     ::[None, 26, 26, 32]::320            ::input_1            ",
                "...............................................................................................",
                "conv2D_2(Conv2D)                     ::[None, 24, 24, 64]::18496          ::conv2D_1           ",
                "...............................................................................................",
                "maxPool2D(MaxPool2D)                 ::[None, 8, 8, 64]  ::0              ::conv2D_2           ",
                "...............................................................................................",
                "conv2D_4(Conv2D)                     ::[None, 8, 8, 64]  ::36928          ::maxPool2D          ",
                "...............................................................................................",
                "conv2D_5(Conv2D)                     ::[None, 8, 8, 64]  ::36928          ::conv2D_4           ",
                "...............................................................................................",
                "add(Add)                             ::[None, 8, 8, 64]  ::0              ::conv2D_5           ",
                "                                     ::                  ::               ::maxPool2D          ",
                "...............................................................................................",
                "conv2D_6(Conv2D)                     ::[None, 8, 8, 64]  ::36928          ::add                ",
                "...............................................................................................",
                "conv2D_7(Conv2D)                     ::[None, 8, 8, 64]  ::36928          ::conv2D_6           ",
                "...............................................................................................",
                "add_1(Add)                           ::[None, 8, 8, 64]  ::0              ::conv2D_7           ",
                "                                     ::                  ::               ::add                ",
                "...............................................................................................",
                "conv2D_8(Conv2D)                     ::[None, 6, 6, 64]  ::36928          ::add_1              ",
                "...............................................................................................",
                "globalAvgPool2D(GlobalAvgPool2D)     ::[None, 64]        ::0              ::conv2D_8           ",
                "...............................................................................................",
                "dense_1(Dense)                       ::[None, 256]       ::16640          ::globalAvgPool2D    ",
                "...............................................................................................",
                "dense_2(Dense)                       ::[None, 10]        ::2570           ::dense_1            ",
                "...............................................................................................",
                ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",
                "Total trainable params: 2570",
                "Total frozen params: 220096",
                "Total params: 222666",
                "..............................................................................................."
            ),
            functionalModel.customFormat(
                layerNameColumnName = "Name of the layer along with its type",
                outputShapeColumnName = "Output shape",
                paramsCountColumnName = "# of parameters",
                connectedToColumnName = "Inbound connections",
                columnSeparator = "::",
                lineSeparatorSymbol = '.',
                thickLineSeparatorSymbol = ':',
            )
        )
    }
}