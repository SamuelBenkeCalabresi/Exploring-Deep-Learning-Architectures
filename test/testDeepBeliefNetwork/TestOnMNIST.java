package testDeepBeliefNetwork;

import dataset.MNISTParser;
import generativeGraphicalModels.DeepBeliefNetwork;
import util.Utils;

public class TestOnMNIST {
	private static void testOnMnist() {
		// 1. import MNIST dataset
		int[][] trainSet = MNISTParser.getDataset("train-images.idx3-ubyte", true);
		int[] trainLabels = MNISTParser.getLabels("train-labels.idx1-ubyte");
		int[][] oneHotEncodedTrainLabels = Utils.getBinaryLabels(trainLabels);
		int[][] testSet = MNISTParser.getDataset("t10k-images.idx3-ubyte", true);
		int[] testLabels = MNISTParser.getLabels("t10k-labels.idx1-ubyte");
		
		// 2. Training deep belief net
		// Define Hyper-Parameters
		int sizeInputLayer = 784;
		int[] hiddenLayerSizes = {300};
		int sizeOutputLayer = 10;		
		int datasetSize = 60000;  
		
		DeepBeliefNetwork dbn = new DeepBeliefNetwork(sizeInputLayer, hiddenLayerSizes, sizeOutputLayer, datasetSize);
		
		// pre-training hyper-parameters + pretrain
		double pretrainLearningRate = 0.1;
		int pretrainingMaxEpoch = 1;
		int gibbsSteps = 1;
		dbn.pretrain(trainSet, pretrainLearningRate, pretrainingMaxEpoch, gibbsSteps);
		
		// fine-tuning hyper-parameters + finetune
		double finetuneLearningRate = 0.1;
		int finetuneMaxEpoch = 1;
		/* HERE THERE ARE MULTIPLE FINE-TUNING METHODS THAT CAN BE USED :
		 * ad default there is the softmax fine-tuning */
		dbn.finetune(trainSet, oneHotEncodedTrainLabels, finetuneLearningRate, finetuneMaxEpoch);
        // dbn.finetune3(trainSet, oneHotEncodedTrainLabels, finetuneLearningRate, finetuneMaxEpoch);

		// 3. Testing deep belief net
		int counter = 0;
		for(int i = 0; i < testSet.length; i++) {
			int prediction = dbn.predict(testSet[i]);
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
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		System.out.println("Test with DBN on MNIST has started...\n");
		testOnMnist();
		long endTime   = System.currentTimeMillis();
		System.out.println("Took: " + ((endTime - startTime) / 1000) + " seconds.");
		System.out.println("Took: " + (((endTime - startTime) / 1000)/60) + " minutes.");
	}
}
