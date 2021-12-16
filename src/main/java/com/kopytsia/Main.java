package com.kopytsia;

import org.apache.commons.math3.linear.RealMatrix;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.kopytsia.Utils.*;
import static java.lang.ClassLoader.getSystemResource;
import static org.apache.commons.math3.linear.MatrixUtils.createColumnRealMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

public class Main {

    private final static int INODES = 784;
    private final static int HNODES = 200;
    private final static int ONODES = 10;
    private final static int EPOCHS = 10;
    private final static double LEARNING_RATE = 0.1;

    public static RealMatrix wInputHidden;
    public static RealMatrix wHiddenOutput;

    public static void main(String[] args) throws Exception {
        wInputHidden = createRealMatrix(HNODES, INODES);
        wHiddenOutput = createRealMatrix(ONODES, HNODES);
        wInputHidden = initRandom(wInputHidden, Math.pow(INODES, -0.5));
        wHiddenOutput = initRandom(wHiddenOutput, Math.pow(HNODES, -0.5));

        int[] labels = Reader.getLabels(Paths.get(getSystemResource("images/train-labels-idx1-ubyte.gz").toURI()));
        List<int[][]> images = Reader.getImages(Paths.get(getSystemResource("images/train-images-idx3-ubyte.gz").toURI()));

        double[][] scaledImages = new double[images.size()][];
        for (int i = 0; i < images.size(); i++) {
            scaledImages[i] = scale(Utils.flat(images.get(i)));
        }

        double[][] roated1ScaledImages = new double[images.size()][];
        for (int i = 0; i < images.size(); i++) {
            roated1ScaledImages[i] = scale(Utils.flat(rotate(images.get(i), 10)));
        }

        double[][] roated2scaledImages = new double[images.size()][];
        for (int i = 0; i < images.size(); i++) {
            roated2scaledImages[i] = scale(Utils.flat(rotate(images.get(i), -10)));
        }

        for (int e = 0; e < EPOCHS; e++) {
            System.out.println("running epoch: " + (e + 1));
            for (int i = 0; i < labels.length; i++) {
                RealMatrix inputs = createColumnRealMatrix(scaledImages[i]);
                RealMatrix targets = createTarget(labels[i]);
                train(inputs, targets);

                inputs = createColumnRealMatrix(roated1ScaledImages[i]);
                train(inputs, targets);

                inputs = createColumnRealMatrix(roated2scaledImages[i]);
                train(inputs, targets);
            }
        }

        int[] testLabels = Reader.getLabels(Paths.get(getSystemResource("images/t10k-labels-idx1-ubyte.gz").toURI()));
        List<int[][]> testImages = Reader.getImages(Paths.get(getSystemResource("images/t10k-images-idx3-ubyte.gz").toURI()));

        int correct = 0;
        for (int i = 0; i < testLabels.length; i++) {
            int correctLabel = testLabels[i];
            RealMatrix predict = query(scale(Utils.flat(testImages.get(i))));
            int predictLabel = indexMax(predict);

            if (predictLabel == correctLabel) {
                correct++;
            }
        }

        System.out.println("Accuracy: " + correct / (double) testLabels.length);

        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(getSystemResource("weights").getFile()))) {
            oos.writeObject(wInputHidden.getData());
            oos.writeObject(wHiddenOutput.getData());
        }

        writeJson(wInputHidden, getSystemResource("js/src/assets/weights-input-hidden.json").getPath());
        writeJson(wHiddenOutput, getSystemResource("js/src/assets/weights-hidden-output.json").getPath());
    }

    private static void writeJson(RealMatrix matrix, String path) throws IOException {
        Path weightsJson = Paths.get(path);
        List<String> output = new ArrayList<>();
        output.add("[");
        double[][] data = matrix.getData();
        for (int r = 0; r < data.length; r++) {
            output.add(Arrays.toString(data[r]));
            if (r < data.length - 1) {
                output.add(",");
            }
        }
        output.add("]");
        Files.write(weightsJson, output);
    }

    public static int indexMax(RealMatrix result) {
        double[][] data = result.getData();
        int indexMax = 0;
        for (int r = 0; r < data.length; r++) {
            indexMax = data[r][0] > data[indexMax][0] ? r : indexMax;
        }

        return indexMax;
    }

    public static RealMatrix query(double[] inputArray) {
        RealMatrix inputs = createColumnRealMatrix(inputArray);
        RealMatrix hiddenInputs = wInputHidden.multiply(inputs);
        RealMatrix hiddenOutputs = scalar(hiddenInputs, Utils::sigmoid);

        RealMatrix finalInputs = wHiddenOutput.multiply(hiddenOutputs);
        RealMatrix finalOutputs = scalar(finalInputs, Utils::sigmoid);
        return finalOutputs;
    }

    private static void train(RealMatrix inputs, RealMatrix targets) {
        // forward
        RealMatrix hiddenInputs = wInputHidden.multiply(inputs);
        RealMatrix hiddenOutputs = scalar(hiddenInputs, Utils::sigmoid);

        RealMatrix finalInputs = wHiddenOutput.multiply(hiddenOutputs);
        RealMatrix finalOutputs = scalar(finalInputs, Utils::sigmoid);

        // back
        RealMatrix outputErrors = targets.subtract(finalOutputs);
        RealMatrix t1 = multiplyElements(outputErrors, finalOutputs);
        RealMatrix t2 = multiplyElements(t1, scalar(finalOutputs, in -> 1.0 - in));
        RealMatrix t3 = t2.multiply(hiddenOutputs.transpose());
        wHiddenOutput = wHiddenOutput.add(scalar(t3, in -> LEARNING_RATE * in));

        RealMatrix hiddenErrors = wHiddenOutput.transpose().multiply(outputErrors);
        t1 = multiplyElements(hiddenErrors, hiddenOutputs);
        t2 = multiplyElements(t1, scalar(hiddenOutputs, in -> 1.0 - in));
        t3 = t2.multiply(inputs.transpose());
        wInputHidden = wInputHidden.add(scalar(t3, in -> LEARNING_RATE * in));
    }

    public static double[] scale(int[] img) {
        double[] result = new double[img.length];
        for (int i = 0; i < img.length; i++) {
            result[i] = img[i] / 255.0 * 0.999 + 0.001;
        }
        return result;
    }

    private static RealMatrix createTarget(int label) {
        RealMatrix target = createRealMatrix(10, 1);
        for (int i = 0; i < 10; i++) {
            target.setEntry(i, 0, i != label ? 0.001 : 0.999);
        }
        return target;
    }

    public static int[][] rotate(int[][] img, double angleInDegrees) {
        double angle = Math.toRadians(angleInDegrees);
        int[][] result = new int[img.length][];
        for (int y = 0; y < img.length; y++) {
            result[y] = new int[img[y].length];
            Arrays.fill(result[y], 0);
        }

        double cosAngle = Math.cos(angle);
        double sinAngle = Math.sin(angle);

        double x0 = img[0].length / 2 - cosAngle * img[0].length / 2
                - sinAngle * img.length / 2;
        double y0 = img.length / 2 - cosAngle * img.length / 2
                + sinAngle * img[0].length / 2;

        for (int y = 0; y < img.length; y++) {
            for (int x = 0; x < img[y].length; x++) {

                int xRot = (int) (x * cosAngle + y * sinAngle + x0);
                int yRot = (int) (-x * sinAngle + y * cosAngle + y0);

                if (xRot >= 0 && yRot >= 0 && xRot <= 27 && yRot <= 27) {
                    result[y][x] = img[yRot][xRot];
                }
            }
        }
        return result;
    }
}
