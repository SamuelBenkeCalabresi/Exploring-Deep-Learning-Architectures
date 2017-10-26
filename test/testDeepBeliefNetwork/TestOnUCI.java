package testDeepBeliefNetwork;

import static util.Utils.getDataLabelsAsInt;
import dataset.BinaryData;
import dataset.UCIParser;
import generativeGraphicalModels.DeepBeliefNetwork;
import util.Utils;

public class TestOnUCI {

	private static void testOnUCI() throws Exception {
		// 1. get data as BinaryData instances
		BinaryData[] trainingDataset = UCIParser.parse("cw2DataSet1.csv");
		BinaryData[] testingDataset  = UCIParser.parse("cw2DataSet2.csv");

		int[][] trainingData = new int[trainingDataset.length][BinaryData.SIZE];
		int[] trainingLabels =  new int[trainingDataset.length];
		trainingLabels = getDataLabelsAsInt(trainingDataset);
		int[][] oneHotEncodedTrainLabels = Utils.getBinaryLabels(trainingLabels);
		int[][] testingData = new int[testingDataset.length][BinaryData.SIZE];
		
		for(int i = 0; i < trainingData.length; i++) {
			trainingData[i] = trainingDataset[i].getData();
			testingData[i] = testingDataset[i].getData();
		}
		
		// Define Hyper-Parameters
		int sizeInputLayer = 64;
		int[] hiddenLayerSizes = {32,32,64};
		int sizeOutputLayer = 10;
		int datasetSize = 2810;  
		// Initialise DBN
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(sizeInputLayer, hiddenLayerSizes, sizeOutputLayer, datasetSize);
		// Training DBN
		double pretrainLearningRate = 0.1;
		int pretrainingMaxEpoch = 100;
		int gibbsSteps = 1;
		dbn.pretrain(trainingData, pretrainLearningRate, pretrainingMaxEpoch, gibbsSteps);
		double finetuneLearningRate = 0.1;
		int finetuneMaxEpochs = 100;
//		double[][] trainingDataAsDouble = getDatasetAsDouble(trainingData);
		dbn.finetune3(trainingData, oneHotEncodedTrainLabels, finetuneLearningRate, finetuneMaxEpochs);
		
		// Check if the weights changes between finetune/finetune3

//		Utils.printMatrix(dbn.softmaxLayer.weights);
//		System.out.println(dbn.softmaxLayer.weights);
		// compare just the softmax layer

		
		int counter = 0;
		for(int i = 0; i < testingData.length; i ++) {
			int prediction = dbn.predict(testingData[i]);

			if(prediction == testingDataset[i].getSolution()) {
				counter ++;
			}
//			else {
//				System.out.println("prediction: " + prediction + ", actual: " + testingDataset[i].getSolution());
//			}
		}
		double accuracyOnTestingSet = (double) counter / testingData.length;
		System.out.println("Testing set size: " + testingData.length);
		System.out.println("Classes predicted right: " + counter);
		System.out.println("Accuracy on the testing set: " + accuracyOnTestingSet * 100 + " %\n");
	}

	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		System.out.println("Test with DBN being performed...\n");
		testOnUCI();
		long endTime   = System.currentTimeMillis();
		System.out.println("Took: " + ((endTime - startTime) / 1000) + " seconds.");
		System.out.println("Took: " + (((endTime - startTime) / 1000)/60) + " minutes.");
	}
}
