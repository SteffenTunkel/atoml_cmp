package mlcrossref.docker;


import mlcrossref.docker.modules.ModelClassifierIris;
import mlcrossref.docker.modules.ModelClassifierWine;
import mlcrossref.docker.modules.ModelGenerator;
import weka.core.FastVector;
import weka.core.Instances;

import weka.classifiers.bayes.NaiveBayes;
import mlcrossref.docker.modules.CSVtoARFF;
import weka.classifiers.functions.Logistic;
import weka.classifiers.evaluation.Evaluation;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.util.ArrayList;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;

import mlcrossref.docker.csvwriter.CSVUtils;
import java.util.ArrayList;
import java.util.List;
import java.io.File;  // Import the File class
import java.io.IOException;  // Import the IOException class to handle errors
import java.io.FileWriter;   // Import the FileWriter class


public class Main {

    //public static final String DATASETPATH = "data/iris.2D.arff";
    //public static final String MODElPATH = "model/model.bin";
    public static final boolean CONVERT = true;

    public static void main(String[] args) throws Exception {

        ModelGenerator mg = new ModelGenerator();



        //String arff_path = conv.convert("data/iris2d.csv");
        // Instances dataset = mg.loadDataset(arff_path);

        //String arff_path_train = conv.convert("data/iris2d_train.csv");
        //String arff_path_test = conv.convert("data/iris2d_test.csv");
        //Instances traindataset = mg.loadDataset(arff_path_train);
        //Instances testdataset = mg.loadDataset(arff_path_test);

        Instances traindataset;
        Instances testdataset;
        if (CONVERT == true) {
            // read data
            CSVtoARFF conv = new CSVtoARFF();
            //conv.convert_split("data/iris/iris_train.csv", "data/iris/iris_test.csv");
            conv.convert_split("/data/weka_train.csv", "/data/weka_test.csv");
            traindataset = mg.loadDataset(conv.train_arff_path);
            testdataset = mg.loadDataset(conv.test_arff_path);
        } else {
            traindataset = mg.loadDataset("/data/weka_train.arff");
            testdataset = mg.loadDataset("/data/weka_test.arff");
        }


        Logistic clf = (Logistic) mg.buildClassifier(traindataset);

        // Evaluate classifier with test dataset
        String evalsummary = mg.evaluateModel(clf, traindataset, testdataset);
        System.out.println("Evaluation: " + evalsummary);

        //Save model
        //mg.saveModel(clf, MODElPATH);

        // save predictions
        ArrayList pred = mg.getPredictions(clf, traindataset, testdataset);


        //System.out.println(pred.get(0));



        // create a csv file
        try {
            File myObj = new File("/log/weka_pred.csv");
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        // write in csv
        try {
            FileWriter myWriter = new FileWriter("/log/weka_pred.csv");
            // write header
            myWriter.write("type,actual,prediction,weigth,dist_0,dist_1\n");
            // write predictions
            for (Object i : pred) {
                // put instance in csv like string (split by ',')
                myWriter.write(i.toString().replace(" ", ",") + '\n');
            };
            myWriter.close();
            System.out.println("Successfully wrote to the file.");

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }


    }

}