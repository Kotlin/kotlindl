package util;

import org.tensorflow.Operand;


public class Triada<T extends Number> {

    private final Operand<T> labels;
    private final Operand<T> lossesOrPredictions;
    private final Operand<T> sampleWeights;

    public Triada(
            Operand<T> labels,
            Operand<T> lossesOrPredictions) {
        this(labels, lossesOrPredictions, null);
    }

    public Triada(
            Operand<T> labels,
            Operand<T> lossesOrPredictions,
            Operand<T> sampleWeights) {
        this.labels = labels;
        this.lossesOrPredictions = lossesOrPredictions;
        this.sampleWeights = sampleWeights;

    }


    public boolean containsLabels() {
        return this.labels != null;
    }

    public boolean containsPredictions() {
        return this.lossesOrPredictions != null;
    }


    public boolean containsLosses() {
        return this.lossesOrPredictions != null;
    }


    public boolean containsSampleWeights() {
        return this.sampleWeights != null;
    }


    public Operand<T> getLabels() {
        return labels;
    }


    public Operand<T> getPredictions() {
        return lossesOrPredictions;
    }


    public Operand<T> getLosses() {
        return lossesOrPredictions;
    }


    public Operand<T> getSampleWeights() {
        return sampleWeights;
    }
}
