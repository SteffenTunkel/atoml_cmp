package mlcrossref.docker.modules;


import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;


public class ModelClassifierIris {

    private Attribute petallength;
    private Attribute petalwidth;
    private Attribute sepallength;
    private Attribute sepalwidth;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classVal;
    private Instances dataRaw;


    public ModelClassifierIris() {
        petallength = new Attribute("petallength");
        petalwidth = new Attribute("petalwidth");
        sepallength = new Attribute("sepallength");
        sepalwidth = new Attribute("sepalwidth");
        attributes = new ArrayList<Attribute>();
        classVal = new ArrayList<String>();
        classVal.add("Iris-setosa");
        classVal.add("Iris-versicolor");
        classVal.add("Iris-virginica");

        attributes.add(petallength);
        attributes.add(petalwidth);
        attributes.add(sepallength);
        attributes.add(sepalwidth);

        attributes.add(new Attribute("class", classVal));
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }

    
    public Instances createInstance(double petallength, double petalwidth, double sepallength, double sepalwidth, double result) {
        dataRaw.clear();
        double[] instanceValue1 = new double[]{petallength, petalwidth, sepallength, sepalwidth, 0};
        dataRaw.add(new DenseInstance(1.0, instanceValue1));
        return dataRaw;
    }


    public String classifiy(Instances insts, String path) {
        String result = "Not classified!!";
        Classifier cls = null;
        try {
            //cls = (MultilayerPerceptron) SerializationHelper.read(path);
            cls = (NaiveBayes) SerializationHelper.read(path);
            result = classVal.get((int) cls.classifyInstance(insts.firstInstance()));
        } catch (Exception ex) {
            Logger.getLogger(ModelClassifierIris.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }


    public Instances getInstance() {
        return dataRaw;
    }
    

}
