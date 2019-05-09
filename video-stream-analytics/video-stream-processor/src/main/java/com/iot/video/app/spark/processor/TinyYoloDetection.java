package com.iot.video.app.spark.processor;

import com.iot.video.app.spark.processor.*;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class TinyYoloDetection {

    private static final double DETECTION_THRESHOLD = 0.5;
    //more accurate but slower
    //less accurate but faster
    public static  ComputationGraph TINY_YOLO_V2_MODEL_PRE_TRAINED;

    private final Stack<Frame> stack = new Stack();
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


    public void warmUp(Speed selectedSpeed, Frame imageMat) throws IOException {
        try {
            Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) TINY_YOLO_V2_MODEL_PRE_TRAINED.getOutputLayer(0);
            //BufferedImage read = ImageIO.read(new File("AutonomousDriving/src/main/resources/sample.jpg"));
            INDArray indArray = prepareImage(imageMat, selectedSpeed.width, selectedSpeed.height);
            INDArray results = TINY_YOLO_V2_MODEL_PRE_TRAINED.outputSingle(indArray);
            outputLayer.getPredictedObjects(results, DETECTION_THRESHOLD);

        } catch (IOException e) {
            System.out.println("Failed to warm , ignoring for now");
        }
    }

    public void push(Frame mat) {
        stack.push(mat);
        System.out.println("sadasd"+stack.size());
    }

    public void drawBoundingBoxesRectangles(Frame frame, Mat matFrame) {
        if (invalidData(frame, matFrame)) return;

        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);
        YoloUtils.nms(detectedObjects, 0.5);
        for (DetectedObject detectedObject : detectedObjects) {
            createBoundingBoxRectangle(matFrame, frame.imageWidth, frame.imageHeight, detectedObject);
        }
    }

    private boolean invalidData(Frame frame, Mat matFrame) {
        return predictedObjects == null || matFrame == null || frame == null;
    }

    public void predictBoundingBoxes(Frame frame) throws IOException {
        long start = System.currentTimeMillis();
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) TINY_YOLO_V2_MODEL_PRE_TRAINED.getOutputLayer(0);
        INDArray indArray = prepareImage(frame, selectedSpeed.width, selectedSpeed.height);
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

    private INDArray prepareImage(Frame frame, int width, int height) throws IOException {
        if (frame == null || frame.image == null) {
            return null;
        }
        BufferedImage convert = new Java2DFrameConverter().convert(frame);
        return prepareImage(convert, width, height);
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
        rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
        putText(file, groupMap.get(map.get(predictedClass)), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
    }

    private final String[] TINY_COCO_CLASSES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};


}