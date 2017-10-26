package generativeGraphicalModels;

import static util.ActivationFunction.*;
import static util.Utils.*;
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
public class RestrictedBoltzmannMachine {
	/* HyperParameters */
	/* Neural network architecture attributes */
	public final int SIZE_VISIBLE_LAYER;
	public final int SIZE_HIDDEN_LAYER;
	public double[][] weights;
	/* Biases as vectors of visibleLayer and hiddenLayer */
	public double[] biasOfHiddenLayer;
	public double[] biasOfVisibleLayer;
	/* Attributes can be useful to have */
	private double energy;
	private Random randomInitializer;

	/* constructor of RestrictedBoltzmannMachine
	 * made mainly to be in the DBN but can be used also in DRBMs or DBMs */ 
	public RestrictedBoltzmannMachine(int numberOfVisibleUnits, int numberOfHiddenUnits, Random randomInitializer) {

		SIZE_VISIBLE_LAYER = numberOfVisibleUnits;
		SIZE_HIDDEN_LAYER = numberOfHiddenUnits;
		this.randomInitializer = randomInitializer;
		initLayers();
	}

	/* It returns the reconstruction of the visible layer after the CD learning */ 
	public double[] reconstructVisibleLayer(int[] v) {
		double[] visibleLayerReconstruction = new double[SIZE_VISIBLE_LAYER];
		double[] hiddenActivations = new double[SIZE_HIDDEN_LAYER];
		double logisticActivation;

		for(int hiddenUnit = 0; hiddenUnit < SIZE_HIDDEN_LAYER; hiddenUnit++) {
			hiddenActivations[hiddenUnit] = stepForward(v, weights[hiddenUnit], biasOfHiddenLayer[hiddenUnit]);
		}
		for(int visibleUnit = 0; visibleUnit < SIZE_VISIBLE_LAYER; visibleUnit++) {
			logisticActivation = 0;

			for(int hiddenUnit = 0; hiddenUnit < SIZE_HIDDEN_LAYER; hiddenUnit++) {
				logisticActivation += weights[hiddenUnit][visibleUnit] * hiddenActivations[hiddenUnit];
			}
			logisticActivation += biasOfVisibleLayer[visibleUnit];
			visibleLayerReconstruction[visibleUnit] = sigmoid(logisticActivation);
		}
		return visibleLayerReconstruction;
	}

	/* It samples the vector in input e.g. a forward step in MLP */
	public void samplingHiddenLayer(int[] input, int[] sample) {

		for(int hiddenUnit = 0; hiddenUnit < SIZE_HIDDEN_LAYER; hiddenUnit++) {			
			for(int visibleUnit = 0; visibleUnit < SIZE_VISIBLE_LAYER; visibleUnit++) {
				sample[hiddenUnit] += weights[hiddenUnit][visibleUnit] * input[visibleUnit];
			}
			sample[hiddenUnit] += biasOfHiddenLayer[hiddenUnit];
			sample[hiddenUnit] = fireBernoulliState(1, sigmoid(sample[hiddenUnit]), randomInitializer);
		}
	}
	/* This can be put in another class
	 * because perform the backprorpagation algorithm of one hidden layer
	 * therefore is not part of the RBM algorithm
	 * however to try to be more efficient was placed in the same class here */
	public double[] backward(int[] layerInput, int[] previousLayerActivations, double[] deltaPreviousLayer, double[][] weightPreviousLayer, double learningRate) {
		double[] deltas = new double[SIZE_HIDDEN_LAYER];

		for(int i=0; i < SIZE_HIDDEN_LAYER; i++) {
			deltas[i] = 0;
			for(int deltaError = 0; deltaError < deltaPreviousLayer.length; deltaError++) {
				deltas[i] += deltaPreviousLayer[deltaError] * weightPreviousLayer[deltaError][i];
			}

			deltas[i] *= dsigmoid(previousLayerActivations[i]);
		}
		for(int hiddenUnits = 0; hiddenUnits < SIZE_HIDDEN_LAYER; hiddenUnits++) {
			for(int visibleUnit = 0; visibleUnit< SIZE_VISIBLE_LAYER; visibleUnit++) {
				weights[hiddenUnits][visibleUnit] += learningRate * deltas[hiddenUnits] * layerInput[visibleUnit];
			}
			biasOfHiddenLayer[hiddenUnits] += learningRate * deltas[hiddenUnits];
		}
		return deltas;
	}

	/*
	 * Free energy as defined in
	 * http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf
	 */
	public double freeEnergy(int[] vector) {
		double sumOverVisibleUnits = 0;
		double sumOverHiddenUnits = 0;
		double sumOverWeights = 0;

		for (int indexVisibleL = 0; indexVisibleL < vector.length; indexVisibleL++) {
			sumOverVisibleUnits += vector[indexVisibleL] * biasOfVisibleLayer[indexVisibleL];

			for (int indexHiddenL = 0; indexHiddenL < SIZE_HIDDEN_LAYER; indexHiddenL++) {
				sumOverWeights += vector[indexVisibleL] * weights[indexHiddenL][indexVisibleL];
				sumOverHiddenUnits += Math.log(1 / + Math.exp(sumOverWeights));
			}
		}
		energy = -sumOverVisibleUnits - sumOverHiddenUnits;
		return energy;
	}
	/*
	 * Free energy as standard in Hopfield networks, as defined in
	 * http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf
	 */
	public double freeEnergy(int[] visibleLayer, int[] hiddenLayer) {
		double sumOverVisibleUnits = 0;
		double sumOverHiddenUnits = 0;
		double sumOverWeights = 0;

		for (int indexVisibleL = 0; indexVisibleL < visibleLayer.length; indexVisibleL++) {
			sumOverVisibleUnits += visibleLayer[indexVisibleL] * biasOfVisibleLayer[indexVisibleL];

			for (int indexHiddenL = 0; indexHiddenL < hiddenLayer.length; indexHiddenL++) {
				sumOverHiddenUnits += hiddenLayer[indexHiddenL] * biasOfHiddenLayer[indexHiddenL];
				sumOverWeights += visibleLayer[indexVisibleL] * hiddenLayer[indexHiddenL]
						* weights[indexHiddenL][indexVisibleL];
			}
		}
		energy = -sumOverVisibleUnits - sumOverHiddenUnits - sumOverWeights;
		return energy;
	}
	/* Contrastive divergence learning algorithm:
	 * It performs the rules as defined in the report.
	 * In the update of the parameters there are commented
	 * the different updates that can be made on the weights
	 * and visible and hidden layers.
	 *  */
	public void contrastiveDivergence(int[] inputVector, double learningRate, int gibbsSteps) {
		double[] positiveHiddenMean = new double[SIZE_HIDDEN_LAYER];
		int[] positiveHiddenSample = new int[SIZE_HIDDEN_LAYER];

		double[] negativeVisibleMeans = new double[SIZE_VISIBLE_LAYER];
		int[] negativeVisibleSamples = new int[SIZE_VISIBLE_LAYER];

		double[] negativeHiddenMeans = new double[SIZE_HIDDEN_LAYER];
		int[] negativeHiddenSamples = new int[SIZE_HIDDEN_LAYER];

		sampleHiddenLayer(inputVector, positiveHiddenMean, positiveHiddenSample);

		for(int step = 0; step < gibbsSteps; step++) {
			if(step == 0) {
				gibbsSampling(positiveHiddenSample, negativeVisibleMeans, negativeVisibleSamples, negativeHiddenMeans, negativeHiddenSamples);
			} else {
				gibbsSampling(negativeHiddenSamples, negativeVisibleMeans, negativeVisibleSamples, negativeHiddenMeans, negativeHiddenSamples);
			}
		}
		for(int hiddenUnits = 0; hiddenUnits < SIZE_HIDDEN_LAYER; hiddenUnits++) {
			for(int visibleUnit = 0; visibleUnit < SIZE_VISIBLE_LAYER; visibleUnit++) {
				weights[hiddenUnits][visibleUnit] += learningRate * (positiveHiddenSample[hiddenUnits] * inputVector[visibleUnit] - (negativeHiddenMeans[hiddenUnits] * negativeVisibleSamples[visibleUnit]));
				//				weights[hiddenUnits][visibleUnit] += learningRate * (positiveHiddenMean[hiddenUnits] * inputVector[visibleUnit] - (negativeHiddenMeans[hiddenUnits] * negativeVisibleSamples[visibleUnit]));
			}
			biasOfHiddenLayer[hiddenUnits] += learningRate * (positiveHiddenSample[hiddenUnits] - negativeHiddenMeans[hiddenUnits]);
			//			biasOfHiddenLayer[hiddenUnits] += learningRate * (positiveHiddenMean[hiddenUnits] - negativeHiddenMeans[hiddenUnits]);
		}
		for(int visibleUnit = 0; visibleUnit < SIZE_VISIBLE_LAYER; visibleUnit++) {
			biasOfVisibleLayer[visibleUnit] += learningRate * (inputVector[visibleUnit] - negativeVisibleSamples[visibleUnit]);
		}
	}
	/* private hidden sampling method used in the Gibbs steps & in the CD contrastive divergence procedure */
	private void sampleHiddenLayer(int[] visibleLayerSample, double[] meanActivation, int[] sample) {
		for(int unit = 0; unit < SIZE_HIDDEN_LAYER; unit++) {
			meanActivation[unit] = stepForward(visibleLayerSample, weights[unit], biasOfHiddenLayer[unit]);
			sample[unit] = fireBernoulliState(1, meanActivation[unit], randomInitializer);
		}
	}
	/* private visible sampling method used in the Gibbs steps & contrastive divergence procedure */
	private void sampleVisibleLayer(int[] hiddenLayersample, double[] meanActivation, int[] sample) {
		for(int unit = 0; unit < SIZE_VISIBLE_LAYER; unit++) {
			meanActivation[unit] = stepBackward(hiddenLayersample, unit, biasOfVisibleLayer[unit]);
			sample[unit] = fireBernoulliState(1, meanActivation[unit], randomInitializer);
		}
	}
	/* private Gibbs samples used how many times the number of the parameter is set.
	 * It s used in the contrastive divergence learning alogrithm */
	private void gibbsSampling(int[] hiddenLayersample, double[] negativeVisibleMeans, 
			int[] negativeVisibleSamples, double[] negativeHiddenMeans, int[] negativeHiddenSamples) {
		sampleVisibleLayer(hiddenLayersample, negativeVisibleMeans, negativeVisibleSamples);
		sampleHiddenLayer(negativeVisibleSamples, negativeHiddenMeans, negativeHiddenSamples);
	}
	/* private step forward to obtain one hidden activation (one neuron) from visible layer/input */
	private double stepForward(int[] inputVector, double[] weight, double bias) {
		double activation = 0;
		for(int visibleUnit = 0; visibleUnit < SIZE_VISIBLE_LAYER; visibleUnit++) {
			activation += weight[visibleUnit] * inputVector[visibleUnit];
		}
		activation += bias;
		return sigmoid(activation);
	}
	/* private step forward to obtain one visible activation (one neuron) from hidden layer */
	private double stepBackward(int[] hiddenLayer, int i, double bias) {
		double activation = 0;
		for(int hiddenUnit = 0; hiddenUnit < SIZE_HIDDEN_LAYER; hiddenUnit++) {
			activation += weights[hiddenUnit][i] * hiddenLayer[hiddenUnit];
		}
		activation += bias;
		return sigmoid(activation);
	}
	/* fireBernoulli state to obtain the samples from the activations:
	 * i.e. to obtain one state of the of one neuron/unit in the RBM */
	private int fireBernoulliState(int n, double activity, Random randomInitializer) {
		int binaryUnit = 0;	
		for(int i = 0; i < n; i++) {
			if (randomInitializer.nextDouble() < activity) {
				binaryUnit++;
			}
		}
		return binaryUnit;
	}
	/* It is used to initialise the weights */ 
	private void initLayers() {
		this.weights = new double[this.SIZE_HIDDEN_LAYER][this.SIZE_VISIBLE_LAYER];
		double valueToRandomize = 1.0 / this.SIZE_VISIBLE_LAYER;

		for(int i=0; i<this.SIZE_HIDDEN_LAYER; i++) {
			for(int j=0; j<this.SIZE_VISIBLE_LAYER; j++) {
				this.weights[i][j] = uniformlyDistributedRandom(-valueToRandomize, valueToRandomize, randomInitializer);
			}
		}
		this.biasOfHiddenLayer = new double[this.SIZE_HIDDEN_LAYER];
		this.biasOfVisibleLayer = new double[this.SIZE_VISIBLE_LAYER];
	}
	/* It is used to make good random starting weights */ 
	private double uniformlyDistributedRandom(double minimumVal, double maximumVal, Random randomInitializer) {
		return randomInitializer.nextDouble() * (maximumVal - minimumVal) + minimumVal;
	}

}
