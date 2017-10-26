package generativeGraphicalModels;

import static util.ActivationFunction.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import artificialNeuralNetworks.AbstractNeuralNetwork;
import artificialNeuralNetworks.SoftmaxRegression;
/**
 * What should be the features of this class: MODEL
 * <br>
 * ARCHITECTURE: (layers) One way to do it:
 * 1. RBM (input vector, feature extractor 1)                -> 784 x 512
 * 2. RBM (output feature extractor 1, feature extractor 2)  -> 512 x 512
 * 3. MLP (output feature extractor 2, deep layer, Softmax output) -> 512 x 2000 x 10
 * <br>
 * LEARNING ALGORITHMS:
 * 1. PreTraining
 * 2. FineTuning (can be applied different tuning algorithms)
 * */ 
public class DeepBeliefNetwork extends AbstractNeuralNetwork {
	/* Define hyper-parameters*/
	private int[] SIZE_HIDDEN_LAYERS;
	private int NUMBER_OF_HIDDEN_LAYERS;
	public RestrictedBoltzmannMachine[] rbmLayers;
	public SoftmaxRegression softmaxLayer;

	public DeepBeliefNetwork(int inputLayerSize, int[] sizesOfHiddenLayers, int outputLayerSize, int datasetSize) {
		super(inputLayerSize, outputLayerSize, datasetSize);

		System.out.print("Init phase has started..");
		SIZE_HIDDEN_LAYERS = sizesOfHiddenLayers;
		NUMBER_OF_HIDDEN_LAYERS = sizesOfHiddenLayers.length;
		rbmLayers = new RestrictedBoltzmannMachine[NUMBER_OF_HIDDEN_LAYERS];
		Random randomInitializer = new Random(1); 

		for(int layer = 0; layer  < this.NUMBER_OF_HIDDEN_LAYERS; layer++) {
			
			int sizeInputOfHiddenLayer;
			if(layer == 0) {
				sizeInputOfHiddenLayer = this.SIZE_INPUT_LAYER;
			} else {
				sizeInputOfHiddenLayer = this.SIZE_HIDDEN_LAYERS[layer-1];
			}
			this.rbmLayers[layer] = new RestrictedBoltzmannMachine(sizeInputOfHiddenLayer, SIZE_HIDDEN_LAYERS[layer], randomInitializer);	
		}
		this.softmaxLayer = new SoftmaxRegression(SIZE_HIDDEN_LAYERS[NUMBER_OF_HIDDEN_LAYERS-1], SIZE_OUTPUT_LAYER, datasetSize);
		System.out.println("done.");
	}

	public void pretrain(int[][] trainingData, double learningRate, int maxEpoch, int gibbsSteps) {
		System.out.print("Pretrain phase has started..");

		for(int i = 0; i < NUMBER_OF_HIDDEN_LAYERS; i++) { 
			
			for(int epoch = 0; epoch < maxEpoch; epoch++) { 
				
				for(int image = 0; image < datasetSize; image++) {  
					int[] inputImageVector = new int[0];
					
					for(int layer = 0; layer <= i; layer++) {	
						int[] previousLayerInput;
						int inputSizePreviousLayer;

						if(layer == 0) {
							inputImageVector = new int[SIZE_INPUT_LAYER];	
							for(int pixel = 0; pixel < SIZE_INPUT_LAYER; pixel++) {
								inputImageVector[pixel] = trainingData[image][pixel];
							} 
							
						} else {
							if(layer == 1) {
								inputSizePreviousLayer = SIZE_INPUT_LAYER;
							} else { 
								inputSizePreviousLayer = SIZE_HIDDEN_LAYERS[layer-2];
							}
							
							previousLayerInput = new int[inputSizePreviousLayer];
							for(int j = 0; j < inputSizePreviousLayer; j++) {
								previousLayerInput[j] = inputImageVector[j];
							} 

							inputImageVector = new int[SIZE_HIDDEN_LAYERS[layer-1]];
							rbmLayers[layer-1].samplingHiddenLayer(previousLayerInput, inputImageVector);
						}
					}
					rbmLayers[i].contrastiveDivergence(inputImageVector, learningRate, gibbsSteps);
				}
			}
		}
		System.out.println(" done.");
	}
	/* This is the fine-tuning which train only the last layer:
	 * SOFTMAX LAYER  with integer input data */
	public void finetune(int[][] trainingData, int[][] trainingLabels, double learningRate, int maxEpochs) {
		
		System.out.print("Finetune phase has started..");
		int[] inputLayer = new int[0];
		int[] previousLayerActivations = new int[0];

		for(int epoch = 0; epoch < maxEpochs; epoch++) {
			for(int image = 0; image < datasetSize; image++) {
				for(int layer = 0; layer < NUMBER_OF_HIDDEN_LAYERS; layer++) {
					if(layer == 0) {
						previousLayerActivations = new int[SIZE_INPUT_LAYER];
						for(int pixel = 0; pixel < SIZE_INPUT_LAYER; pixel++) {
							previousLayerActivations[pixel] = trainingData[image][pixel];
						} 
					} else {
						previousLayerActivations = new int[SIZE_HIDDEN_LAYERS[layer-1]];
						for(int pixel = 0; pixel < SIZE_HIDDEN_LAYERS[layer-1]; pixel++) {
							previousLayerActivations[pixel] = inputLayer[pixel];	
						}
					}

					inputLayer = new int[SIZE_HIDDEN_LAYERS[layer]];
					rbmLayers[layer].samplingHiddenLayer(previousLayerActivations, inputLayer);
				}
				softmaxLayer.train(inputLayer, trainingLabels[image], learningRate);
			}
		}
		System.out.println(" done.");
	}
	/* This is the fine-tuning which train only the last layer:
	 * SOFTMAX LAYER with double trainig data. */
	/* NB: not used with MNIST */
	public void finetune2(double[][] trainingData, int[][] trainingLabels, double learningRate, int maxEpochs) {
		System.out.print("Finetune phase has started..");
		double[] inputLayer = new double[0];
		double[] previousLayerActivations = new double[0];

		for(int epoch = 0; epoch < maxEpochs; epoch++) {
			for(int image = 0; image < datasetSize; image++) {
				for(int layer = 0; layer < NUMBER_OF_HIDDEN_LAYERS; layer++) {
					
					if(layer == 0) {
						previousLayerActivations = new double[SIZE_INPUT_LAYER];
						for(int pixel = 0; pixel < SIZE_INPUT_LAYER; pixel++) {
							previousLayerActivations[pixel] = trainingData[image][pixel];
						} 
					} else {
						previousLayerActivations = new double[SIZE_HIDDEN_LAYERS[layer-1]];
						for(int pixel = 0; pixel < SIZE_HIDDEN_LAYERS[layer-1]; pixel++) {
							previousLayerActivations[pixel] = inputLayer[pixel];
						} 
					}
					inputLayer = new double[SIZE_HIDDEN_LAYERS[layer]];
				}
				softmaxLayer.train(inputLayer, trainingLabels[image], learningRate);
			}
		}
		System.out.println(" done.");
	}
	
	/* It deep backpropagates to all the layers:
	 * WHOLE BACKWARD PASS
	 * NB: this can be changed to make it how 
	 * many backprop layers are required */
	public void finetune3(int[][] trainingData, int[][] trainingLabels, double learningRate, int maxEpochs) {
		System.out.print("Finetune phase has started..");
		/* FORWARD PASS: storing the samples output of each layer */
		for(int epoch = 0; epoch < maxEpochs; epoch++) {
			for(int image = 0; image < datasetSize; image ++) {
				
				int[] inputLayer = new int[0];
				int[] previousLayerActivations = new int[0];
				double[] deltaY;
				ArrayList<int[]> layerOutputs = new ArrayList<>(NUMBER_OF_HIDDEN_LAYERS+1); 
				layerOutputs.add(trainingData[image]);

				for(int layer = 0; layer < NUMBER_OF_HIDDEN_LAYERS; layer++) {
					if(layer == 0) {
						previousLayerActivations = new int[SIZE_INPUT_LAYER];
						for(int pixel = 0; pixel < SIZE_INPUT_LAYER; pixel++) {
							previousLayerActivations[pixel] = trainingData[image][pixel];
						} 
						
					} else {
						previousLayerActivations = new int[SIZE_HIDDEN_LAYERS[layer-1]];
						for(int pixel = 0; pixel < SIZE_HIDDEN_LAYERS[layer-1]; pixel++) {
							previousLayerActivations[pixel] = inputLayer[pixel];
						} 
					}

					inputLayer = new int[SIZE_HIDDEN_LAYERS[layer]];
					rbmLayers[layer].samplingHiddenLayer(previousLayerActivations, inputLayer);
					layerOutputs.add(inputLayer.clone());
				}
				deltaY = softmaxLayer.train(inputLayer, trainingLabels[image], learningRate);
				/* BACKWARD PASS: updating the paramters of each layer */
				double[][] previousWeights;
				double[] newDelta = new double[0];
	
				for (int layer = NUMBER_OF_HIDDEN_LAYERS - 1; layer >= 0; layer--) {
					if (layer == NUMBER_OF_HIDDEN_LAYERS - 1) {
						previousWeights = softmaxLayer.weights;
					} else {
						previousWeights = rbmLayers[layer+1].weights;
						deltaY = newDelta.clone();
					}
					newDelta = rbmLayers[layer].backward(layerOutputs.get(layer), layerOutputs.get(layer+1), deltaY, previousWeights, learningRate);
				}
			}
		}
		System.out.println(" done.");
	}
	/* It returns an integer (0/9) of the class predicted */
	public int predict(int[] inputVector) {  
		
		double[] y = new double [SIZE_OUTPUT_LAYER];
		double[] inputLayer = new double[0];
		double[] previousLayerActivations = new double[SIZE_INPUT_LAYER];
		double output;
		
		for(int pixel = 0; pixel < SIZE_INPUT_LAYER; pixel++) {
			previousLayerActivations[pixel] = inputVector[pixel];	
		}
		// FORWARD PASS
		for(int layer = 0; layer < NUMBER_OF_HIDDEN_LAYERS; layer++) {
			
			inputLayer = new double[rbmLayers[layer].SIZE_HIDDEN_LAYER];
			for(int rbmHiddenUnit = 0; rbmHiddenUnit < rbmLayers[layer].SIZE_HIDDEN_LAYER; rbmHiddenUnit++) {
				output = 0;
				for(int rbmVisibleUnits = 0; rbmVisibleUnits < rbmLayers[layer].SIZE_VISIBLE_LAYER; rbmVisibleUnits++) {
					output += rbmLayers[layer].weights[rbmHiddenUnit][rbmVisibleUnits] * previousLayerActivations[rbmVisibleUnits];
				}
				output += rbmLayers[layer].biasOfHiddenLayer[rbmHiddenUnit];
				inputLayer[rbmHiddenUnit] = sigmoid(output);
			}
			if(layer < NUMBER_OF_HIDDEN_LAYERS-1) {
				previousLayerActivations = new double[rbmLayers[layer].SIZE_HIDDEN_LAYER];
				for(int pixel = 0; pixel < rbmLayers[layer].SIZE_HIDDEN_LAYER; pixel++) {
					previousLayerActivations[pixel] = inputLayer[pixel];
				} 
			}
		}
		// SOFTMAX REGRESSION
		for(int softmaxOuputUnit = 0; softmaxOuputUnit < softmaxLayer.SIZE_OUTPUT_LAYER; softmaxOuputUnit++) {
			y[softmaxOuputUnit] = 0;
			for(int softmaxInputUnit = 0; softmaxInputUnit < softmaxLayer.SIZE_INPUT_LAYER; softmaxInputUnit++) {
				y[softmaxOuputUnit] += softmaxLayer.weights[softmaxOuputUnit][softmaxInputUnit] * inputLayer[softmaxInputUnit];
			}
			y[softmaxOuputUnit] += softmaxLayer.bias[softmaxOuputUnit];
		}
		softmaxLayer.softmax(y);
		return argmax(y);
	}
}