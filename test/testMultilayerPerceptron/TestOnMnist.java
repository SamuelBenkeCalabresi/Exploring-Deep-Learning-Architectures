package testMultilayerPerceptron;

import artificialNeuralNetworks.MultilayerPerceptron;
import dataset.MNISTParser;
import util.Utils;

public class TestOnMnist {

	private static void testOnMnist() {
		// 1. import dataset
		int[][] trainSet = MNISTParser.getDataset("train-images.idx3-ubyte", false);
		int[] trainLabels = MNISTParser.getLabels("train-labels.idx1-ubyte");
		int[][] oneHotEncodedTrainLabels = Utils.getBinaryLabels(trainLabels);
		int[][] testSet = MNISTParser.getDataset("t10k-images.idx3-ubyte", false);
		int[] testLabels = MNISTParser.getLabels("t10k-labels.idx1-ubyte");

		// 2. Training MLP classifier
		// Define Hyper-Parameters
		int sizeInputLayer = 784;
		int sizeHiddenLayer = 200;   
		int sizeOutputLayer = 10;
		double learningRate = 0.1; 		
		int maxEpochs = 1; 		    
		int trainDataSetSize = 60000;  

		// Initialise MLP
		MultilayerPerceptron mlp = new MultilayerPerceptron(sizeInputLayer, sizeHiddenLayer, sizeOutputLayer,
				learningRate, maxEpochs, trainDataSetSize);
		// Fitting the MLP
		mlp.fit(trainSet, oneHotEncodedTrainLabels);

		// 3. Testing MLP classifier
		int counter = 0;
		for(int i = 0; i < testSet.length; i++) {
			int prediction = mlp.predict(testSet[i]);
			if(prediction == testLabels[i]) {
				counter ++;
			}
			else {
				System.out.println("prediction class: " + prediction);
				System.out.println("real class: " + testLabels[i]);
			}
		}
		double accuracyOnTestingSet = (double) counter / testSet.length;
		System.out.println("Testing set size: " + testSet.length);
		System.out.println("Classes predicted right: " + counter);
		System.out.println("Accuracy on the testing set: " + accuracyOnTestingSet * 100 + " %\n");
	}
	public static void main(String[]  args) {
		long startTime = System.currentTimeMillis();
		System.out.println("Test with MLP clf on MNIST has started...\n");
		testOnMnist();
		long endTime   = System.currentTimeMillis();
		System.out.println("Took: " + ((endTime - startTime) / 1000) + " seconds.");
		System.out.println("Took: " + (((endTime - startTime) / 1000)/60) + " minutes.");
	}
}
