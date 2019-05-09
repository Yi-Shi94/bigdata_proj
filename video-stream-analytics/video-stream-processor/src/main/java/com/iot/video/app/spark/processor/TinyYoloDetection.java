package com.iot.video.app.spark.processor;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

import org.opencv.core.*;

import static org.opencv.core.Core.FONT_HERSHEY_DUPLEX;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;

public class TinyYoloDetection {

    private static final double DETECTION_THRESHOLD = 0.5;
    public static  ComputationGraph TINY_YOLO_V2_MODEL_PRE_TRAINED;
    private final Stack<INDArray> stack = new Stack();
    private final Speed selectedSpeed;
    private volatile List<DetectedObject> predictedObjects;
    private HashMap<Integer, String> map;
    private HashMap<String, String> groupMap;

    static {
        try {
            TINY_YOLO_V2_MODEL_PRE_TRAINED = (ComputationGraph) TinyYOLO.builder().build().initPretrained();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public  TinyYoloDetection(Speed selectedSpeed) throws IOException {
        this.selectedSpeed = selectedSpeed;
        prepareTinyYOLOLabels();
    }

    public void warmUp(Speed selectedSpeed, INDArray imageArray) throws IOException {
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) TINY_YOLO_V2_MODEL_PRE_TRAINED.getOutputLayer(0);
        //BufferedImage read = ImageIO.read(new File("AutonomousDriving/src/main/resources/sample.jpg"));
        //INDArray indArray = prepareImage(imageArray, selectedSpeed.width, selectedSpeed.height);
        INDArray results = TINY_YOLO_V2_MODEL_PRE_TRAINED.outputSingle(imageArray);
        outputLayer.getPredictedObjects(results, DETECTION_THRESHOLD);
    }

    public void push(INDArray arr) {
        stack.push(arr);
        System.out.println("sadasd"+stack.size());
    }

    public void drawBoundingBoxesRectangles(Mat matFrame) {
        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);
        YoloUtils.nms(detectedObjects, 0.5);
        for (DetectedObject detectedObject : detectedObjects) {
            createBoundingBoxRectangle(matFrame, matFrame.cols(), matFrame.rows(), detectedObject);
        }
    }

    public List<DetectedObject> getPredectionOfCurrentImage(){
        return predictedObjects;
    }


    public void predictBoundingBoxes(INDArray indArray) throws IOException {
        long start = System.currentTimeMillis();
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) TINY_YOLO_V2_MODEL_PRE_TRAINED.getOutputLayer(0);
        System.out.println("stack of frames size " + stack.size());
        if (indArray == null) {
            return;
        }
        INDArray results = TINY_YOLO_V2_MODEL_PRE_TRAINED.outputSingle(indArray);
        if (results == null) {
            return;
        }
        predictedObjects = outputLayer.getPredictedObjects(results, DETECTION_THRESHOLD);
        System.out.println("stack of predictions size " + predictedObjects.size());
        System.out.println("Prediction time " + (System.currentTimeMillis() - start) / 1000d);
    }

    public boolean existPredictionBox(){
        if(predictedObjects.size()>0) return true;
        else return false;
    }


    private boolean invalidData( Mat matFrame) {
        return predictedObjects == null || matFrame == null;
    }

    private INDArray prepareImage(BufferedImage convert, int width, int height) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);

        INDArray indArray = loader.asMatrix(convert);
        if (indArray == null) {
            return null;
        }
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }


    private void prepareLabels(String[] coco_classes) {
        if (map == null) {
            groupMap = new HashMap<>();
            groupMap.put("car", "Car");
            groupMap.put("bus", "Car");
            groupMap.put("truck", "Car");
            int i = 0;
            map = new HashMap<>();
            for (String s1 : coco_classes) {
                map.put(i++, s1);
                groupMap.putIfAbsent(s1, s1);
            }
        }
    }

    private void prepareTinyYOLOLabels() {
        prepareLabels(TINY_COCO_CLASSES);
    }

    private void createBoundingBoxRectangle(Mat file, int w, int h, DetectedObject obj) {
        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();
        int predictedClass = obj.getPredictedClass();
        int x1 = (int) Math.round(w * xy1[0] / selectedSpeed.gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / selectedSpeed.gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / selectedSpeed.gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / selectedSpeed.gridHeight);
        rectangle(file, new Point(x1,y1), new Point(x2,y2),new Scalar(0,255,0));
        putText(file, groupMap.get(map.get(predictedClass)), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1.0,new Scalar(255,0,0) );
    }

    private final String[] TINY_COCO_CLASSES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

}