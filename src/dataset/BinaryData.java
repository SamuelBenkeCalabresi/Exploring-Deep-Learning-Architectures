package dataset;

public class BinaryData extends AbstractDataset{
	/* The data on any row of the matrix */
	// NB change them all to integers
	
	/* The possible solution or class label the data belongs */
	private final double solution;
	/* one-hot encoding giving 1 to class label of such data and 0 to all others */
	/* It is used in probability approaches were required */
	private final double[] binarySolution;
		
	/* Size of the row */
	public static final int SIZE = 64;
	/* number of solutions */
	public static final int SOLUTION_MAX_VALUE = 9;
	public static final int SOLUTION_LENGTH = 10;
	/* labelled data */
	public static final int SIZE_LABELLED_DATA = SIZE + SOLUTION_MAX_VALUE + 1;
	
	/* Constructor of Binary Data */
	public BinaryData(int [] data, double solution) {
		super(data);
		this.solution = solution;
		this.binarySolution = initBinarySolution(solution);
	}
	
	public double[] getBinarySolution() {
		return binarySolution;
	}

	public int[] getData() {
		return data;
	}

	public double getSolution() {
		return solution;
	}
	
	/* It returns labelled data as array of doubles */
	public double[] getLabelledData() {
		double[] labelledData = new double[SIZE_LABELLED_DATA];
		for(int i = 0; i < SIZE; i++) {
			labelledData[i] = data[i];
		}
		int j = 0;
		for(int i = SIZE; i < SIZE_LABELLED_DATA; i++) {
			labelledData[i] = binarySolution[j];
			j++;
		}	
		return labelledData;
	}
	
	public static double[][] getLabelledData(BinaryData [] binaryDataSet) {
		double [][] labelledDataSet = new double [binaryDataSet.length][SIZE_LABELLED_DATA];
		for(int i = 0; i < binaryDataSet.length; i++) {
			labelledDataSet[i] = binaryDataSet[i].getLabelledData();
	    }
		return labelledDataSet;
	}
	
	public static int [][] getUnlabelledDataAsInt(BinaryData [] trainingDataSet) {
		int [][] unlabelledDataSet = new int [trainingDataSet.length][SIZE];
		for(int i = 0; i < trainingDataSet.length; i++) {
//			System.out.println(trainingDataSet[i].toString());
			unlabelledDataSet[i] = trainingDataSet[i].getData();
//			System.out.println(unlabelledDataSet[i]);
	    }
		return unlabelledDataSet;
	}
	
	
	@Override
	public String toString() {
		String output = "Binary data: ";
		for(double element : data) {
			output += element + ", ";
		}
		output += "\n";
		output += "Solution:    " + solution + "\n";
		output += "Binary solution: ";
		for(double element : binarySolution) {
			output += element + ", ";
		}
		return output;
	}
	
	public static double[] initBinarySolution(double solution) {
		double[] binarySolution = new double[SOLUTION_MAX_VALUE+1];
				
		for(int i = 0; i < binarySolution.length; i++) {
			/* if the indexed element is == solution number, 
			 * set that index element to 1, all zeros the others */
			if(i != solution) {
				binarySolution[i] = 0;
			}
			else {
				binarySolution[i] = 1;
			}
		}
		return binarySolution;
	}
	
	public int[] returnBinarySolutionsAsInt(int solution) {
		int[] binarySolution = new int[SOLUTION_LENGTH];
				
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
	
}
