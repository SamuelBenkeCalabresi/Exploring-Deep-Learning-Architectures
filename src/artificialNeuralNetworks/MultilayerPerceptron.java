package artificialNeuralNetworks;

import static util.ActivationFunction.*;
import java.util.Random;

/**
 * It has been used in the project to 
 * compare it to the DBN.
 * <br>
 * Model of MultilayerPerceptron (MLP):
 * ARCHITECTURE: (layers)
 * 1. input layer  (input vector) 
 * 2. hidden layer (non-linear transformation)
 * 3. output layer (number of classes to predict from) 
 * <br>
 * LEARNING ALGORTIHM: 
 * 1. get activations of inputs
 * 2. get activations of hidden layer
 * 3. get outputs
 * 4. get cost function/deltas difference between actual class and outputs
 * 5. get gradients/value for update the parameters
 * 6. update weights and biases
 * */
public class MultilayerPerceptron extends AbstractNeuralNetwork{
	/* Define Hyper-Parameters */
	/* Neural network architecture attributes */
	private final int SIZE_HIDDEN_LAYER;
	private double weightsOfHiddenLayer[][];
	private double weightsOfOutputLayer[][];
	/* Bias as vectors of visibleLayer and hiddenLayer */
	private double[] biasOfHiddenLayer;
	private double[] biasOfOutputLayer;
	/* Numerical Meta-Parameters */
	public int testDatasetSize;

	public MultilayerPerceptron(int sizeInputLayer, int sizeHiddenLayer, int sizeOutputLayer, double learningRate, int epochs, int datasetSize) {
		super(sizeInputLayer, sizeOutputLayer, datasetSize);
		this.learningRate = learningRate;
		this.maxEpoch = epochs; 
		SIZE_HIDDEN_LAYER = sizeHiddenLayer;
		Random randomInit = new Random(1);
		this.randomInitializer = randomInit;
		initLayers();
	}

	/**
	 * As standard: get X, Y training data
	 * @param trainingData   
	 * @param trainingLabels        
	 */
	public void fit(int[][] trainingData, int[][] trainingLabels) {
		for(int epoch = 0; epoch < maxEpoch; epoch ++) {
			for(int i = 0; i < trainingData.length; i++) {
				train(trainingData[i], trainingLabels[i]);
			}
		}
	}
	
	/**
	 * As standard: get testingData
	 * @param testingData   
	 */
	public int predict(int[] testingData) {
		// HIDDEN LAYER
		double[] hiddenActivations = new double[SIZE_HIDDEN_LAYER];
		
		for(int indexHidden = 0; indexHidden < SIZE_HIDDEN_LAYER; indexHidden++) {
			hiddenActivations[indexHidden] = 0;
			for(int indexInput = 0; indexInput < SIZE_INPUT_LAYER; indexInput++) {
				hiddenActivations[indexHidden] += weightsOfHiddenLayer[indexHidden][indexInput] * testingData[indexInput];
			}	
			hiddenActivations[indexHidden] += biasOfHiddenLayer[indexHidden];
			hiddenActivations[indexHidden] = tanh(hiddenActivations[indexHidden]);
		}
		// OUTPUT LAYER 
		double[] predictedLabel = new double[testingData.length];

		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			predictedLabel[i] = 0;

			for(int j = 0; j < SIZE_HIDDEN_LAYER; j++) {
				predictedLabel[i] += weightsOfOutputLayer[i][j] * hiddenActivations[j];
			}
			predictedLabel[i] += biasOfOutputLayer[i]; 
		}
		softmax(predictedLabel);
		int solution = argmax(predictedLabel);
		return solution;
	}
	
	/* It return the whole softmax probabilities instead of only the class predicted */
	public double[] predictSoft(double[] testingData) {

		double[] hiddenActivations = new double[SIZE_HIDDEN_LAYER];
		double[] hiddenActivations2 = new double[SIZE_HIDDEN_LAYER];
		for(int indexHidden = 0; indexHidden < SIZE_HIDDEN_LAYER; indexHidden++) {
			hiddenActivations[indexHidden] = 0;
			for(int indexInput = 0; indexInput < SIZE_INPUT_LAYER; indexInput++) {
				hiddenActivations[indexHidden] += weightsOfHiddenLayer[indexHidden][indexInput] * testingData[indexInput];
			}	
			hiddenActivations[indexHidden] += biasOfHiddenLayer[indexHidden];
			hiddenActivations2[indexHidden] = tanh(hiddenActivations[indexHidden]);
		}
		double[] outputActivations = new double[SIZE_OUTPUT_LAYER];
		for(int indexOuput = 0; indexOuput < SIZE_OUTPUT_LAYER; indexOuput++) {
			outputActivations[indexOuput] = 0;
			for(int indexHidden = 0; indexHidden  < SIZE_HIDDEN_LAYER; indexHidden ++) {
				outputActivations[indexOuput] += weightsOfOutputLayer[indexOuput][indexHidden] * hiddenActivations2[indexHidden];
			}	
			outputActivations[indexOuput] += biasOfOutputLayer[indexOuput];
		}
		softmax(outputActivations);	
		return outputActivations;
	}
	
	/* Backpropagation learning algorithm */
	private void train(int[] trainingData, int[] trainingLabel) {
		// forward Propagation
		double[] activationsOfHiddenLayer =  new double[SIZE_HIDDEN_LAYER];
		double[] outputActivations = new double[SIZE_OUTPUT_LAYER];
		forwardPropagation(trainingData, activationsOfHiddenLayer, outputActivations);
		// backward Propagation
		backwardPropagation(trainingData, trainingLabel, outputActivations, activationsOfHiddenLayer);
	}

	/* Should be the activations hiddenLayer, activations outputLayer (using activations hiddenLayer) */
	private void forwardPropagation(int[] trainingData, double[] activationsOfHiddenLayer, double[] outputActivations) {
		double[] hiddenActivations =  new double[SIZE_HIDDEN_LAYER];

		for(int indexHidden = 0; indexHidden < SIZE_HIDDEN_LAYER; indexHidden++) {
			hiddenActivations[indexHidden] = 0;
			for(int indexInput = 0; indexInput < SIZE_INPUT_LAYER; indexInput++) {
				hiddenActivations[indexHidden] += weightsOfHiddenLayer[indexHidden][indexInput] * trainingData[indexInput];
			}	
			hiddenActivations[indexHidden] += biasOfHiddenLayer[indexHidden];
//			// We then call the activation function 
			hiddenActivations[indexHidden] = tanh(hiddenActivations[indexHidden]);
			activationsOfHiddenLayer[indexHidden] = hiddenActivations[indexHidden];

		}
				
		for(int indexOuput = 0; indexOuput < SIZE_OUTPUT_LAYER; indexOuput++) {
			outputActivations[indexOuput] = 0;
			for(int indexHidden = 0; indexHidden  < SIZE_HIDDEN_LAYER; indexHidden ++) {
				outputActivations[indexOuput] += weightsOfOutputLayer[indexOuput][indexHidden] * hiddenActivations[indexHidden];
			}	
			outputActivations[indexOuput] += biasOfOutputLayer[indexOuput];
		}
		softmax(outputActivations);	
	}

	private void backwardPropagation(int[] trainingData, int[] trainingLabel, double[] outputActivations, double[] hiddenActiv) {
		/* Find the error at the output units */
		// Create delta 
		double[] delta = new double[SIZE_OUTPUT_LAYER];
		// Update OutputLayer weights
		for(int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER; indexOutput++) {
			// We define delta as the difference between the real label and the softmax probability
			delta[indexOutput] = trainingLabel[indexOutput] - outputActivations[indexOutput];

			for(int indexHidden = 0; indexHidden < SIZE_HIDDEN_LAYER; indexHidden++) {
				weightsOfOutputLayer[indexOutput][indexHidden] += learningRate * delta[indexOutput] * hiddenActiv[indexHidden] / datasetSize;
			}	
			biasOfOutputLayer[indexOutput] += learningRate * delta[indexOutput] / datasetSize;
		}	
		// Create delta2  
		double[] delta2 = new double[SIZE_HIDDEN_LAYER];
		for(int i = 0; i < SIZE_HIDDEN_LAYER; i++) {
			delta2[i] = 0;
			for(int j = 0; j < SIZE_OUTPUT_LAYER; j++) {
				// Why is it multiplying the weights for delta1?
				delta2[i] += delta[j] * weightsOfOutputLayer[j][i];
			}
			delta2[i] *= dtanh(hiddenActiv[i]);
		}
		// Update HiddenLayer weights
		for(int i = 0; i < SIZE_HIDDEN_LAYER; i++) {
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
                weightsOfHiddenLayer[i][j] += learningRate * delta2[i] * (double)trainingData[j] / datasetSize;
			}
            biasOfHiddenLayer[i] += learningRate * delta2[i] / datasetSize;
		}	
	}
	/* It initialise the weights */
	public void initLayers() {
		/* Initialise the weights */
	    Random rng = new Random(1);
        double distributeRandom = 1.0 / SIZE_INPUT_LAYER;
		weightsOfHiddenLayer = new double[SIZE_HIDDEN_LAYER][SIZE_INPUT_LAYER]; 
		weightsOfOutputLayer = new double[SIZE_OUTPUT_LAYER][SIZE_HIDDEN_LAYER]; 
		/* Initialise the biases */
		biasOfHiddenLayer = new double[SIZE_HIDDEN_LAYER];
		biasOfOutputLayer = new double[SIZE_OUTPUT_LAYER];		
		for(int i = 0; i < SIZE_HIDDEN_LAYER; i++) {
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				weightsOfHiddenLayer[i][j] = rng.nextDouble() * (distributeRandom - (-distributeRandom)) + (-distributeRandom);
			}
		}
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			for(int j = 0; j < SIZE_HIDDEN_LAYER; j++) {
				weightsOfOutputLayer[i][j] = rng.nextDouble() * (distributeRandom - (-distributeRandom)) + (-distributeRandom);
			}
		}
	}
}
