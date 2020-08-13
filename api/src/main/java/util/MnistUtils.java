package util;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Stream;

public class MnistUtils {
    /**
     * Read random {@code count} samples from MNIST dataset from two files (images and labels) into a stream of labeled
     * vectors.
     *
     * @param imagesPath Path to the file with images.
     * @param labelsPath Path to the file with labels.
     * @param rnd        Random numbers generator.
     * @param cnt        Count of samples to read.
     * @return Stream of MNIST samples.
     */
    public static Stream<double[]> mnistAsStream(String imagesPath, String labelsPath, Random rnd, int cnt)
            throws IOException {
        FileInputStream isImages = new FileInputStream(imagesPath);
        FileInputStream isLabels = new FileInputStream(labelsPath);

        read4Bytes(isImages); // Skip magic number.
        int numOfImages = read4Bytes(isImages);
        int imgHeight = read4Bytes(isImages);
        int imgWidth = read4Bytes(isImages);

        read4Bytes(isLabels); // Skip magic number.
        read4Bytes(isLabels); // Skip number of labels.

        int numOfPixels = imgHeight * imgWidth;

        double[][] vecs = new double[numOfImages][numOfPixels + 1];

        for (int imgNum = 0; imgNum < numOfImages; imgNum++) {
            vecs[imgNum][numOfPixels] = isLabels.read();
            for (int p = 0; p < numOfPixels; p++) {
                int c = 128 - isImages.read();
                vecs[imgNum][p] = (double) c / 128;
            }
        }

        List<double[]> lst = Arrays.asList(vecs);
        Collections.shuffle(lst, rnd);

        isImages.close();
        isLabels.close();

        return lst.subList(0, cnt).stream();
    }

    /**
     * Read random {@code count} samples from MNIST dataset from two files (images and labels) into a stream of labeled
     * vectors.
     *
     * @param imagesPath Path to the file with images.
     * @param labelsPath Path to the file with labels.
     * @param rnd        Random numbers generator.
     * @param cnt        Count of samples to read.
     * @return List of MNIST samples.
     * @throws IOException In case of exception.
     */
    public static List<LabeledImage> mnistAsList(String imagesPath, String labelsPath, Random rnd,
                                                 int cnt) throws IOException {
        return mnistAsList(new FileInputStream(imagesPath), new FileInputStream(labelsPath), rnd, cnt);
    }


    /**
     * Read random {@code count} samples from MNIST dataset from two resources (images and labels) into a stream of
     * labeled vectors.
     *
     * @param imageStream Stream with image data.
     * @param lbStream    Stream with label data.
     * @param rnd         Random numbers generator.
     * @param cnt         Count of samples to read.
     * @return List of MNIST samples.
     * @throws IOException In case of exception.
     */
    private static List<LabeledImage> mnistAsList(InputStream imageStream, InputStream lbStream, Random rnd,
                                                  int cnt) throws IOException {
        List<LabeledImage> res = new ArrayList<>();

        read4Bytes(imageStream); // Skip magic number.
        int numOfImages = read4Bytes(imageStream);
        int imgHeight = read4Bytes(imageStream);
        int imgWidth = read4Bytes(imageStream);

        read4Bytes(lbStream); // Skip magic number.
        read4Bytes(lbStream); // Skip number of labels.

        int numOfPixels = imgHeight * imgWidth;

        for (int imgNum = 0; imgNum < numOfImages; imgNum++) {
            double[] pixels = new double[numOfPixels];
            for (int p = 0; p < numOfPixels; p++)
                pixels[p] = (float) (1.0 * (imageStream.read() & 0xFF) / 255);
            res.add(new LabeledImage(pixels, lbStream.read()));
        }

        Collections.shuffle(res, rnd);

        return res.subList(0, cnt);
    }


    /**
     * Utility method for reading 4 bytes from input stream.
     *
     * @param is Input stream.
     * @throws IOException In case of exception.
     */
    private static int read4Bytes(InputStream is) throws IOException {
        return (is.read() << 24) | (is.read() << 16) | (is.read() << 8) | (is.read());
    }

    /**
     * MNIST image.
     */
    public static class Image {
        /**
         * Pixels.
         */
        private final double[] pixels;

        /**
         * Construct a new instance of MNIST image.
         *
         * @param pixels Pixels.
         */
        public Image(double[] pixels) {
            this.pixels = pixels;
        }

        /**
         *
         */
        public double[] getPixels() {
            return pixels;
        }
    }

    /**
     * MNIST labeled image.
     */
    public static class LabeledImage extends Image {
        /**
         * Label.
         */
        private final int lb;

        /**
         * Constructs a new instance of MNIST labeled image.
         *
         * @param pixels Pixels.
         * @param lb     Label.
         */
        public LabeledImage(double[] pixels, int lb) {
            super(pixels);
            this.lb = lb;
        }

        /**
         *
         */
        public int getLabel() {
            return lb;
        }
    }
}