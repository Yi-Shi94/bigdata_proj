package com.iot.video.app.spark.processor;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.Iterator;


import org.apache.log4j.Logger;

import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.opencv.core.Mat;

import static org.opencv.imgcodecs.Imgcodecs.imwrite;

import com.iot.video.app.spark.util.VideoEventData;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Class to use YOLO detection with Kafka
 *
 * @author Y.Shi 2019
 *
 */

public class VideoObjectDetector implements Serializable {
    private static final Logger logger = Logger.getLogger(VideoMotionDetector.class);

    //load native lib
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public static VideoEventData objectDetect(String camId, Iterator<VideoEventData> frames, String outputDir, VideoEventData previousProcessedEventData) throws Exception {

        //Speed speedSetting = Speed.FAST;    // low accuracy, real-time
        //Speed speedSetting = Speed.MEDIUM;  // Almost Real-time and medium accuracy
        //Speed speedSetting = Speed.SLOW;    // High accuracy, really fucking slow

        Speed speedSetting = Speed.MEDIUM;

        TinyYoloDetection tinyYoloDetection = new TinyYoloDetection(speedSetting);
        VideoEventData currentProcessedEventData = new VideoEventData();
        NativeImageLoader loader = new NativeImageLoader(speedSetting.height, speedSetting.width, 3);

        Mat processedImageMat = null;

        //To Be Implemented
        //Sort by timestamp, in a sorted linked list, I am considering to show image in sequence with display.
        //However, Kafka seems forbidding me imshow during runtime, I am considering a standalone process which start with a display, keep reading images
        //from processed-data folder. Meanwhile Kafka may output image in different order from my indexes, be careful

        //You may do something with MongoDB, one Idea is to use getPredictedObjects, you can get the box top left and bottom right
        //points, store a time stamp and a prediction rectangle postion.

        //You may also try those settings with different number of videos. I recommand to make them extremly short just like the creepy one with pillow and sofa.
        //And add them to the report.

        //Forget about UI, I think the code we have are unusable in our case.

        //If you want, you can use recognizer to read h5 file, to perform recognition of other stuff (sign,text,...).


        ArrayList<VideoEventData> sortedList = new ArrayList<VideoEventData>();

        while(frames.hasNext()){
            sortedList.add(frames.next());
        }

        sortedList.sort(Comparator.comparing(VideoEventData::getTimestamp));

        logger.warn("cameraId="+camId+" total frames="+sortedList.size());
        String id ;
        System.out.println("Finish sorting this batch\n");

        //iterate and detect motion
        int index = 0;

        for (VideoEventData eventData : sortedList) {
            if(index>42) break;
            Mat mat = getMat(eventData);
            INDArray imageArr = loader.asMatrix(mat);

            ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
            imagePreProcessingScaler.transform(imageArr);

            String idInfo = "cameraId=" + camId + "-T-timestamp=" + eventData.getTimestamp().toString();
            logger.warn(idInfo);

            tinyYoloDetection.predictBoundingBoxes(imageArr);
            if (tinyYoloDetection.existPredictionBox()) {
                tinyYoloDetection.drawBoundingBoxesRectangles(mat);
                saveImage(mat, idInfo, outputDir);
            } else {
                System.out.println("No prediction box found in this frame, skipped");
            }
            currentProcessedEventData = eventData;
            index += 1;
        }
        return currentProcessedEventData;
    }

    //Get Mat from byte[]
    private static Mat getMat(VideoEventData ed) throws Exception{
        Mat mat = new Mat(ed.getRows(), ed.getCols(), ed.getType());
        mat.put(0, 0, Base64.getDecoder().decode(ed.getData()));
        return mat;
        //return new Mat(ed.getRows(), ed.getCols(), CV_8UC(ed.getType()), new BytePointer(Base64.getDecoder().decode(ed.getData())));
    }

    //Save image file
    private static void saveImage(Mat mat,String idInfo,String outputDir){
        String imagePath = outputDir+idInfo+".png";
        logger.warn("Saving images to "+imagePath);
        //boolean result = cvSaveImage(imagePath, mat);
        if(mat==null){
            System.out.println("Mat valid");
        }
        boolean result = imwrite(imagePath,mat);
        if(!result ){
            System.out.println("No address valid");
        }
    }


}
