package dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class UCIParser {
	
	static final int DATA_SIZE = 64;
	
	public static ArrayList<String> getDataElements(String file) throws Exception {
		BufferedReader in = null;
		String line;
		ArrayList<String> linesOfDataElements = new ArrayList<>();
		
		try {
			in = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			while ((line = in.readLine()) != null) {
//				System.out.println("line: " + line);
//				System.out.println("line lenght: " + line.length());
				/* getDataElements */
				linesOfDataElements.add(line);
				}
		} catch (NumberFormatException e) {
				throw new Exception();
			}
		return linesOfDataElements;
	}
	
	/* a try to use just double and integers instead of a new data structure, NB: use just to test */
	public static BinaryData[] parse(String file) throws Exception {
		
		ArrayList<String> linesOfDataElements = getDataElements(file);
		final int NUM_DATA_ELEMENTS = linesOfDataElements.size();  
		BinaryData [] dataSet = new BinaryData [NUM_DATA_ELEMENTS];

		String line;
		BufferedReader in = null;

		try {
			in = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		try { 
			getBinaryData(dataSet, linesOfDataElements);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataSet;
	}
	
	private static void getBinaryData(BinaryData [] dataSet, ArrayList<String> linesOfDataElements) throws Exception {

		double solution = 0;
		
		for(int i = 0; i < dataSet.length; i++) {
			
			String line = linesOfDataElements.get(i);
			line = line.trim();
			
			if (!line.equals("")) {
				/* Split the line using the comma */
				String[] tokens = line.split(",");
				int[] data = new int[BinaryData.SIZE]; 

				try {
					for(int j = 0; j < tokens.length; j ++) {			
						/* add all data for binary data */
						/* if the integer is less than 8 is equal 0, 1 otherwise */
						
				    	// PROBLEM: what about the solution?
						/* if the element is not the last, add to unlabelled data set */
						if(j != tokens.length - 1) {
							int currentData = Integer.parseInt(tokens[j]);
							
						    if(currentData > 7) {
						    	currentData = 1;
						    	/* add the zero to dataSet current row */
//						    	dataSet[i][j] = data;
						    	data[j] = currentData;
						    }
						    else {
						    	currentData = 0;
						    	/* add the one to dataSet current row */
//						    	dataSet[i][j] = data;
						    	data[j] = currentData;
//						    	System.out.print("data " + j + " : " + data[j] + ", ");

						    }
						}						
						/* add to solution array otherwise */
						else {
							solution = Integer.parseInt(tokens[j]);
						}
						
						/* add data to the BinaryData data structure */
						dataSet[i] = new BinaryData(data.clone(), solution);
//				    	System.out.println();
					}
				} catch (NumberFormatException e) {
					throw e;
				}
			}
		}
	}
	
	public static void printDataSet(double [][] dataSet) {
		for(double [] row : dataSet) {
			int i = 0;
			for(double element : row) {
				if(i == BinaryData.SIZE+1) {
			    System.out.print(" labels: " + element + ", ");	
			    i++;
				}
				else {
					System.out.print(element + ", ");
					i++;
				}
			}
			System.out.println();
		}
	}
	
	public static void printRowDataSet(double [][] dataSet) {
		for(int i = 0; i < dataSet.length; i++) {
			for(int j = 0; j < dataSet[0].length; j++) {
				System.out.print(dataSet[i][j] + ", ");
			}
			System.out.println();
		}
	}
	
}
