package atoml.testgen;

import java.util.List;

import atoml.metamorphic.MetamorphicTest;

import atoml.smoke.Bias;
import atoml.smoke.Categorical;
import atoml.smoke.DisjointCategorical;
import atoml.smoke.DisjointNumeric;
import atoml.smoke.LeftSkew;
import atoml.smoke.ManyCategories;
import atoml.smoke.MaxDouble;
import atoml.smoke.MaxFloat;
import atoml.smoke.MinDouble;
import atoml.smoke.MinFloat;
import atoml.smoke.OneClass;
import atoml.smoke.Outlier;
import atoml.smoke.RandomCategorial;
import atoml.smoke.RandomNumeric;
import atoml.smoke.RightSkew;
import atoml.smoke.SmokeTest;
import atoml.smoke.Split;
import atoml.smoke.StarvedBinary;
import atoml.smoke.StarvedMany;
import atoml.smoke.Uniform;
import atoml.smoke.VeryLarge;
import atoml.smoke.VerySmall;
import atoml.smoke.Zeroes;
import atoml.smoke.RandomNumericSplit;
import atoml.smoke.UniformSplit;
import atoml.smoke.SmoketestFromArff;

public class TestCatalog {

	/**
	 * List of all smoke tests
	 * Used by atoml_cmp for the test case generation
	 * Alongside the implemented datasets (see imports) also external can be used with:
	 *      new SmoketestFromArff("TestName", "/ArffFile.arff")
	 * In this case the data in the given arff file ist used training and testing.
	 * To use separated training and test sets use:
	 *      new SmoketestFromArff("TestName", "/ArffFileTraining.arff", "/ArffFileTest.arff")
	 */
	public static final List<SmokeTest> SMOKETESTS = List.of(
			new UniformSplit(),
			new RandomNumericSplit(),
	        new SmoketestFromArff("BreastCancer", "/BreastCancer_training.arff", "/BreastCancer_test.arff"));
	/**
	 * Immutable list of all metamorphic tests, needed by atoml but not used by atoml_cmp.
	 */
	public static final List<MetamorphicTest> METAMORPHICTESTS = List.of();
}
