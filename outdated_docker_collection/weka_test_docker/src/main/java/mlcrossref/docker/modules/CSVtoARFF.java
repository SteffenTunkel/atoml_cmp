package mlcrossref.docker.modules;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;


import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;


public class CSVtoARFF {
    public String train_arff_path;
    public String test_arff_path;

    public String convert(String path) throws Exception{
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();

        String arff_path = path.replace(".csv", ".arff");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arff_path));
        saver.writeBatch();
        return arff_path;
    }

    public void convert_split(String train_path, String test_path) throws Exception{
        // load CSVs
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(train_path));
        Instances train_data = loader.getDataSet();
        loader.setSource(new File(test_path));
        Instances test_data = loader.getDataSet();

        // write ARFFs
        this.train_arff_path = train_path.replace(".csv", ".arff");
        this.test_arff_path = test_path.replace(".csv", ".arff");
        ArffSaver saver = new ArffSaver();

        saver.setInstances(train_data);
        saver.setFile(new File(this.train_arff_path));
        saver.writeBatch();

        saver.setInstances(test_data);
        saver.setFile(new File(this.test_arff_path));
        saver.writeBatch();

    }
}

