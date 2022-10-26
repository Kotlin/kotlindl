package org.jetbrains.kotlinx.dl.api.summary


public interface ModelWithSummary {
    /**
     * Returns model summary.
     *
     * @return model summary
     */
    public fun summary(): ModelSummary
}
