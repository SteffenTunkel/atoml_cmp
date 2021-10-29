package atoml.testgen;

import java.util.List;

import atoml.metamorphic.MetamorphicTest;

import atoml.smoke.SmokeTest;
import atoml.smoke.SmoketestFromArff;
import atoml.smoke.RandomNumericSplit;
import atoml.smoke.UniformSplit;

public class TestCatalog {

	/**
	 * List of all tests
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
	        new SmoketestFromArff("BreastCancer", "/BreastCancer_training.arff", "/BreastCancer_test.arff"),
	        new SmoketestFromArff("BreastCancerZNorm", "/BreastCancerZNorm_training.arff", "/BreastCancerZNorm_test.arff"),
	        new SmoketestFromArff("BreastCancerMinMaxNorm", "/BreastCancerMinMaxNorm_training.arff", "/BreastCancerMinMaxNorm_test.arff"),
            new SmoketestFromArff("Wine", "/Wine_training.arff", "/Wine_test.arff"),
            new SmoketestFromArff("WineZNorm", "/WineZNorm_training.arff", "/WineZNorm_test.arff"),
            new SmoketestFromArff("WineMinMaxNorm", "/WineMinMaxNorm_training.arff", "/WineMinMaxNorm_test.arff"));

	/**
	 * Immutable list of all metamorphic tests, needed by atoml but not used by atoml_cmp.
	 */
	public static final List<MetamorphicTest> METAMORPHICTESTS = List.of();
}
