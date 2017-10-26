package artificialNeuralNetworks;
/** 
 * The SoftmaxRegression class implements 
 * the Softmax regression model. 
 * <br>
 * It is a supervised probabilistic model that returns with 
 * predict function, a vector of any class label
 * giving to each label a probability that sum to 1
 */
public class SoftmaxRegression extends AbstractNeuralNetwork{

	/* The constructor has to */
	/* Define the HyperParameters of SoftmaxRegression network */
    /* At least the architectural ones */
    public SoftmaxRegression(int inputLayerSize, int outputLayerSize, int datasetSize) {
    	super(inputLayerSize, outputLayerSize, datasetSize);
    	
        weights = new double[SIZE_OUTPUT_LAYER][SIZE_INPUT_LAYER];
        bias = new double[SIZE_OUTPUT_LAYER];
    }
    
    public SoftmaxRegression(int inputLayerSize, int outputLayerSize, int datasetSize, double learningRate, int maxEpoch) {
    	super(inputLayerSize, outputLayerSize, datasetSize);
    	this.learningRate = learningRate;
    	this.maxEpoch = maxEpoch;

        weights = new double[SIZE_OUTPUT_LAYER][SIZE_INPUT_LAYER];
        bias = new double[SIZE_OUTPUT_LAYER];
    }
    
    /**
	 * 
	 * @param testingData 
	 *            The testing data set type of double because 
	 */
	public int predict(int[] testingData) {
		double[] predictedLabel = new double[testingData.length];
		
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			predictedLabel[i] = 0;
			
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				// Here we're doing the stuff with the weights learned in the train
				predictedLabel[i] += weights[i][j] * testingData[j];
			}
			// Here we're doing the stuff with the bias learned in the train
			predictedLabel[i] += bias[i]; 
		}
		softmax(predictedLabel);
		int solution = argmax(predictedLabel);
		return solution;	
	}
	
	/* This has reason to exist for DBN, in the case where we get the activations
	 * from all the previous RBMs and makes sense to keep them as double and only at the
	 * end softmax and argmax those values */
	public int predict(double[] testingData) {
		double[] predictedLabel = new double[testingData.length];
		
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			predictedLabel[i] = 0;
			
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				// Here we're doing the stuff with the weights learned in the train
				predictedLabel[i] += weights[i][j] * testingData[j];
			}
			// Here we're doing the stuff with the bias learned in the train
			predictedLabel[i] += bias[i]; 
		}
		softmax(predictedLabel);
		return argmax(predictedLabel);	
	}
    
	/* fit method for stand-alone use of a softmax architecture */
	public void fit(int[][] trainingData, int[][] trainingLabels) {
 		for(int epoch = 0; epoch < maxEpoch; epoch ++) {
			for(int i = 0; i < trainingData.length; i++) {
				train(trainingData[i], trainingLabels[i]);
			}
		}
	}
	
	/* training SoftmaxRegression as defined in:
	 * http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/ */
	public double[] train(int[] trainingData, int[] trainingLabel) {
		
		// softmax probabilities
		double[] softmaxProbabilities = new double[SIZE_OUTPUT_LAYER];
		
		// initialise softmaxProbabilities
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			softmaxProbabilities[i] = 0;
			
			// for any unit in the input vector
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				// We define softmaxProbabilities[i] =  multiplication of weight and trainingData value
				softmaxProbabilities[i] += weights[i][j] * trainingData[j];
			}
			// We add the bias, as default = 0
			softmaxProbabilities[i] += bias[i];
		}
		// We then call the softmax activation function;
		softmax(softmaxProbabilities);
		
		// We then perform the weights update
		return updateParameters(trainingData, trainingLabel, softmaxProbabilities, this.learningRate);
	}
	
	/* training SoftmaxRegression as defined in:
	 * http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/ */
	public double[] train(int[] trainingData, int[] trainingLabel, double learningRate) {
		
		// softmax probabilities
		double[] softmaxProbabilities = new double[SIZE_OUTPUT_LAYER];
		
		// initialise softmaxProbabilities
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			softmaxProbabilities[i] = 0;
			
			// for any unit in the input vector
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				// We define softmaxProbabilities[i] =  multiplication of weight and trainingData value
				softmaxProbabilities[i] += weights[i][j] * trainingData[j];
			}
			// We add the bias, as default = 0
			softmaxProbabilities[i] += bias[i];
		}
		// We then call the softmax activation function;
		softmax(softmaxProbabilities);
		
		// We then perform the weights update
		return updateParameters(trainingData, trainingLabel, softmaxProbabilities, learningRate);
	}
	
	/* training SoftmaxRegression as defined in:
	 * http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/ */
	public double[] train(double[] trainingData, int[] trainingLabel, double learningRate) {
		
		// softmax probabilities
		double[] softmaxProbabilities = new double[SIZE_OUTPUT_LAYER];
		
		// initialise softmaxProbabilities
		for(int i = 0; i < SIZE_OUTPUT_LAYER; i++) {
			softmaxProbabilities[i] = 0;
			
			// for any unit in the input vector
			for(int j = 0; j < SIZE_INPUT_LAYER; j++) {
				// We define softmaxProbabilities[i] =  multiplication of weight and trainingData value
				softmaxProbabilities[i] += weights[i][j] * trainingData[j];
			}
			// We add the bias, as default = 0
			softmaxProbabilities[i] += bias[i];
		}
		// We then call the softmax activation function;
		softmax(softmaxProbabilities);
		
		// We then perform the weights update
		return updateParameters(trainingData, trainingLabel, softmaxProbabilities, learningRate);
	}
	
	/**
	 *  updateWeights function concretely
	 *  updates the weights of the network:
	 *  1. multiplying delta for training data 
	 *  2. updating the bias vector
	 *  3. returning delta inside the train function */
	private double[] updateParameters(int[] trainingData, int[] trainingLabel, double[] softmaxProbabilities, double learningRate) {
		// Delta of target Y -> Delta of difference between predicted target and real target
		double[] delta = new double[SIZE_OUTPUT_LAYER];
		
		// Update the weights!
		for(int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER; indexOutput++) {
			// We define delta as the difference between the real label and the softmax probability
			delta[indexOutput] = trainingLabel[indexOutput] - softmaxProbabilities[indexOutput];
			
			for(int indexInput = 0; indexInput < SIZE_INPUT_LAYER; indexInput++) {
				weights[indexOutput][indexInput] += learningRate * delta[indexOutput] * trainingData[indexInput]; ///((double)datasetSize);
			}	
			// Update the bias unit!
			bias[indexOutput] += learningRate * (delta[indexOutput]);// /((double)datasetSize));
		}	
		return delta;
	}	
	
	/**
	 *  updateWeights function concretely
	 *  updates the weights of the network:
	 *  1. multiplying delta for training data 
	 *  2. updating the bias vector
	 *  3. returning delta inside the train function */
	private double[] updateParameters(double[] trainingData, int[] trainingLabel, double[] softmaxProbabilities, double learningRate) {
		// Delta of target Y -> Delta of difference between predicted target and real target
		double[] delta = new double[SIZE_OUTPUT_LAYER];
		
		// Update the weights!
		for(int indexOutput = 0; indexOutput < SIZE_OUTPUT_LAYER; indexOutput++) {
			// We define delta as the difference between the real label and the softmax probability
			delta[indexOutput] = trainingLabel[indexOutput] - softmaxProbabilities[indexOutput];
			
			for(int indexInput = 0; indexInput < SIZE_INPUT_LAYER; indexInput++) {
				weights[indexOutput][indexInput] += learningRate * delta[indexOutput] * trainingData[indexInput] / datasetSize;
			}	
			// Update the bias unit!
			bias[indexOutput] += learningRate * (delta[indexOutput] / datasetSize);
		}	
		return delta;
	}
	
}
