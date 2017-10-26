package artificialNeuralNetworks;

import java.util.Random;
/**
 * 
 * AbstractANN is an abstract class for machine learning
 * neural network models.
 * <br>
 * It contains the most important and common hyper-parameters in 
 * artificial neural networks architectures. 
 * <br>
 */
public abstract class AbstractNeuralNetwork {

	/* Define HyperParameters */
	/* Define constants */
	/* number of units in visible layer, as default in MNIST is 28 x 28 matrix = 784 vector */
	public final int SIZE_INPUT_LAYER;
	/* number of output units, as default for discrimination problem to classify numbers = 10 */
	public final int SIZE_OUTPUT_LAYER;

	/* Weights matrix of synapses between neurons */
	public double[][] weights;
	/* Bias vector */
	public double[] bias;
	/* Learning rate */
	protected double learningRate;
	/* Number of epochs (epochs to loop whole dataset) */ 
	protected int maxEpoch;
	/* Dataset size */
	protected int datasetSize;
	/* To randomly initialise the weights */
	protected Random randomInitializer;

	/* Construct has to initialise the architectural HyperParameters */
	public AbstractNeuralNetwork(int sizeInputLayer, int sizeOutputLayer, int datasetSize) {
		this.SIZE_INPUT_LAYER = sizeInputLayer;
		this.SIZE_OUTPUT_LAYER = sizeOutputLayer;
		this.datasetSize = datasetSize;
	}

	public void softmax(double[] outputProbabilities) {
		double sum = 0;
		double max = 0;

		for (int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER ; indexOutput++) {
			if (max < outputProbabilities[indexOutput]) {
				max = outputProbabilities[indexOutput];  
			}
		}
		for (int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER; indexOutput++) {
			outputProbabilities[indexOutput] = Math.exp(outputProbabilities[indexOutput] - max);
			sum += outputProbabilities[indexOutput];
		}

		for (int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER; indexOutput++) {
			outputProbabilities[indexOutput] /= sum;
		}		
	}

	/* function to one-hot encode the probabilities from softmax method */
	public int argmax(double[] output) {
		int classOfMaximumProbability = Integer.MIN_VALUE;
		double max = Integer.MIN_VALUE;

		for(int indexOutput = 0; indexOutput < output.length; indexOutput++) {
			if(max < output[indexOutput]) {
				max = output[indexOutput];
				classOfMaximumProbability = indexOutput;
			}
		}	
		return classOfMaximumProbability;
	}
	
	protected double uniformlyDistributedRandom(double minimumVal, double maximumVal, Random randomInitializer) {
		return randomInitializer.nextDouble() * (maximumVal - minimumVal) + minimumVal;
	}

}
