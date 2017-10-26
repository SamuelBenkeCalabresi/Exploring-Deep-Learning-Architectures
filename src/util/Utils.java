package util;

import java.util.Random;

import dataset.BinaryData;

public class Utils {
	/* The following methods are just helper methods
	 * that could be inside the Main test class */

	public static void printMatrix(int[][] x) {
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[0].length; j++) {
				System.out.print(x[i][j] + ", ");
			}
			System.out.println();
		}
	}

	public static void printMatrix(double[][] x) {
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[0].length; j++) {
				System.out.print(x[i][j] + ", ");
			}
			System.out.println();
		}
	}

	public static void printArray(int[] x) {
		for(int i = 0; i < x.length; i++) {
			System.out.print(x[i] + ", ");
		}
		System.out.println();
	}

	public static int[][] getBinaryLabels(int[] labelSet) {
		int[][] softmaxLabelSet = new int[labelSet.length][10];

		for(int i = 0; i < softmaxLabelSet.length; i++) {
			softmaxLabelSet[i] = returnBinaryLabel(labelSet[i]);
		}
		return softmaxLabelSet;
	}

	// In this way we use the softmax/argmax labels
	public static int[] returnBinaryLabel(int solution) {
		int[] binarySolution = new int[10];

		for(int i = 0; i < binarySolution.length; i++) {
			/* if the indexed element is == solution number, set that index element to 1, all zeros the others */
			if(i != solution) {
				binarySolution[i] = 0;
			}
			else {
				binarySolution[i] = 1;
			}
		}
		return binarySolution;
	}

	public static double[][] getDatasetAsDouble(int[][] x) {
		double[][] y = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[0].length; j++) {
				y[i][j] = x[i][j];
			}
		}
		return y;
	}

	/* Get data labels Y as binary solution integers */
	public static int[] getDataLabelsAsInt(BinaryData[] dataSet) {
		int[] trainingLabels = new int[dataSet.length];

		for(int i = 0; i < trainingLabels.length; i++) {
			trainingLabels[i] = (int) dataSet[i].getSolution();
		}
		return trainingLabels;
	}
}
