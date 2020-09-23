package api.keras

/** Model saving format. */
enum class ModelFormat {
    /** Saves model as graph in .pb file format and variables in .txt file format. */
    TF_GRAPH_CUSTOM_VARIABLES,

    /** Saves model as graph in .pb file format and variables in .txt file format. */
    TF_GRAPH,

    /** Saves model as a list of layers in .json file format and variables in .txt file format. */
    KERAS_CONFIG_CUSTOM_VARIABLES
}
