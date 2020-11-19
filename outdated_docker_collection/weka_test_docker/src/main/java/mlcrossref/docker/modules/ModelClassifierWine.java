package mlcrossref.docker.modules;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ModelClassifierWine {

    private Attribute fixed_acidity;
    private Attribute volatile_acidity;
    private Attribute citric_acid;
    private Attribute residual_sugar;
    private Attribute chlorides;
    private Attribute free_sulfur_dioxide;
    private Attribute total_sulfur_dioxide;
    private Attribute density;
    private Attribute pH;
    private Attribute sulphates;
    private Attribute alcohol;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classVal;
    private Instances dataRaw;


    public ModelClassifierWine() {
        fixed_acidity = new Attribute("fixed_acidity");
        volatile_acidity = new Attribute("volatile_acidity");
        citric_acid = new Attribute("citric_acid");
        residual_sugar = new Attribute("residual_sugar");
        chlorides = new Attribute("chlorides");
        free_sulfur_dioxide = new Attribute("free_sulfur_dioxide");
        total_sulfur_dioxide = new Attribute("total_sulfur_dioxide");
        density = new Attribute("density");
        pH = new Attribute("pH");
        sulphates = new Attribute("sulphates");
        alcohol = new Attribute("alcohol");

        attributes = new ArrayList<Attribute>();
        classVal = new ArrayList<String>();
        classVal.add("good");
        classVal.add("bad");

        attributes.add(fixed_acidity);
        attributes.add(volatile_acidity);
        attributes.add(citric_acid);
        attributes.add(residual_sugar);
        attributes.add(chlorides);
        attributes.add(free_sulfur_dioxide);
        attributes.add(total_sulfur_dioxide);
        attributes.add(density);
        attributes.add(pH);
        attributes.add(sulphates);
        attributes.add(alcohol);


        attributes.add(new Attribute("class", classVal));
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }


    public Instances createInstance(double fixed_acidity, double volatile_acidity, double citric_acid,
                                    double residual_sugar, double chlorides, double free_sulfur_dioxide,
                                    double total_sulfur_dioxide, double density, double pH, double sulphates,
                                    double alcohol, double result) {
        dataRaw.clear();
        double[] instanceValue1 = new double[]{fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,
                free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol, 0};
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
