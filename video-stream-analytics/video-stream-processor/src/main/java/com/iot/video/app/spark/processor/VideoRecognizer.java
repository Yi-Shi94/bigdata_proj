package com.iot.video.app.spark.processor;

import java.io.File;
import com.iot.video.app.spark.util.VideoEventData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class VideoRecognizer {
    //openCV library
    static {
        nu.pattern.OpenCV.loadLocally();
    }


    private static Logger logger = LoggerFactory.getLogger(VideoRecognizer.class);
    private final String modelDir = "src/main/resources/generatedModels/DL4J/DL4J_Iris_Model.zip";

    public void VideoRecognizer(String[] args) throws Exception {
        File storedTrainedModel = new File(modelDir);
        MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(storedTrainedModel);
        // Inference
        INDArray input = Nd4j.create(new double[] {5.0,3.5,1.6,0.6});
        INDArray result = restoredModel.output(input);
        System.out.println("Probabilities: " + result.toString());
    }

    private static void saveImage(Mat mat, VideoEventData ed, String outputDir){
        String imagePath = outputDir+ed.getCameraId()+"-T-"+ed.getTimestamp().getTime()+".png";
        logger.warn("Saving images to "+imagePath);
        boolean result = Imgcodecs.imwrite(imagePath, mat);
        if(!result){
            logger.error("Couldn't save images to path "+outputDir+".Please check if this path exists. This is configured in processed.output.dir key of property file.");
        }
    }
}