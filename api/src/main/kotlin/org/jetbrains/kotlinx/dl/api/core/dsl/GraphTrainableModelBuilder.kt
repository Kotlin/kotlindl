package org.jetbrains.kotlinx.dl.api.core.dsl

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Layer

@DslMarker
@Retention(AnnotationRetention.SOURCE)
internal annotation class BuilderDsl

@DslMarker
@Retention(AnnotationRetention.SOURCE)
internal annotation class EntryDsl


/**
 * Graph trainable model builder.
 *
 * @param [T] Type of model.
 * @property [createModelFunction] Function accept layers and create the model.
 * @constructor Creates Graph trainable model builder.
 */
public class GraphTrainableModelBuilder<T : GraphTrainableModel>(private val createModelFunction: (Array<Layer>) -> T) {

    /** Model builder block. */
    private var modelBuilderBlock: T.() -> Unit = {}

    /** Model layers. */
    private var layerListBuilderBlock: LayerListBuilder.() -> Unit = {}

    /**
     * Use block.
     *
     * Defaults to null. When null, do nothing after build.
     */
    private var useBlock: ((T) -> Unit)? = null

    /**
     * Model Builder.
     *
     * accept model, for example
     * ```
     *     model {
     *         name = "MyModel"
     *     }
     * ```
     * @param [block] Model builder block.
     */
    @BuilderDsl
    public fun model(block: T.() -> Unit) {
        modelBuilderBlock = block
    }

    /**
     * Layers Builder.
     *
     * @see [LayerListBuilder]
     * @param [block] Layer builder block.
     */
    @BuilderDsl
    public fun layers(block: LayerListBuilder.() -> Unit) {
        layerListBuilderBlock = block
    }

    /**
     * Use builder.
     *
     * Apply after model builder block, will close automatically.
     *
     * @param [block] Use builder block.
     */
    @BuilderDsl
    public fun use(block: T.() -> Unit) {
        useBlock = block
    }

    public fun build(): T = createModelFunction(LayerListBuilder().apply(layerListBuilderBlock).toArray())
        .apply {
            modelBuilderBlock()
            useBlock?.let { use(it) }
        }

}

/**
 * Layer list builder.
 *
 * @property [layers] A mutable list to store layers.
 * @constructor Creates an empty Layer list builder.
 */
@JvmInline
public value class LayerListBuilder(private val layers: MutableList<Layer> = mutableListOf()) {

    /**
     * Unary plus.
     *
     * Add a layer to the list and return itself.
     *
     * We can do this
     * ```
     *     +Input(128, 128)
     * ```
     * instead of
     * ```
     *     layers.add(Input(128, 128))
     * ```
     * @return the layer
     */
    public operator fun Layer.unaryPlus(): Layer {
        layers.add(this)
        return this
    }

    /**
     * Converts layers to array.
     *
     * @return array of layers.
     */
    public fun toArray(): Array<Layer> = layers.toTypedArray()
}

/**
 * Sequential model builder.
 *
 * @param [builder] The builder block.
 * @return a Sequential model.
 */
@EntryDsl
public fun sequential(builder: GraphTrainableModelBuilder<Sequential>.() -> Unit): Sequential {
    return GraphTrainableModelBuilder(Sequential.Companion::of).apply(builder).build()
}

/**
 * Functional model builder.
 *
 * @param [builder] The builder block.
 * @return a Functional model.
 */
@EntryDsl
public fun functional(builder: GraphTrainableModelBuilder<Functional>.() -> Unit): Functional {
    return GraphTrainableModelBuilder(Functional.Companion::of).apply(builder).build()
}

/**
 * Generic model builder.
 *
 * @receiver Function to create model, accepts an array of layer.
 * @param [T] type of model
 * @param [builder] The builder block.
 * @return a model corresponding to [T].
 */
@EntryDsl
public operator fun <T : GraphTrainableModel> ((Array<Layer>) -> T).invoke(builder: GraphTrainableModelBuilder<T>.() -> Unit): T {
    return GraphTrainableModelBuilder(this).apply(builder).build()
}
