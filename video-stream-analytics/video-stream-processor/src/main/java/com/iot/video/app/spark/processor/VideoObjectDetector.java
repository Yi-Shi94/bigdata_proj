package com.iot.video.app.spark.processor;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;
import org.bytedeco.javacpp.BytePointer;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_highgui;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;



/*
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
*/


import com.iot.video.app.spark.util.VideoEventData;


/**
 * Class to detect motion from video frames using OpenCV library.
 *
 * @author abaghel
 *
 */
public class VideoObjectDetector implements Serializable {
    private static final Logger logger = Logger.getLogger(VideoMotionDetector.class);

    //load native lib
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public static VideoEventData objectDetect(String camId, Iterator<VideoEventData> frames, String outputDir, VideoEventData previousProcessedEventData) throws Exception {
        VideoEventData currentProcessedEventData = new VideoEventData();
        Mat processedImageMat = null;

        TinyYoloDetection tinyYoloDetection = new TinyYoloDetection();

        //sort by timestamp
        ArrayList<VideoEventData> sortedList = new ArrayList<VideoEventData>();
        while(frames.hasNext()){
            sortedList.add(frames.next());
        }

        sortedList.sort(Comparator.comparing(VideoEventData::getTimestamp));
        logger.warn("cameraId="+camId+" total frames="+sortedList.size());

        //iterate and detect motion
        int index = 0;
        for (VideoEventData eventData : sortedList) {
            Mat frame = getMat(eventData);
            String idInfo = "cameraId=" + camId + "-T-timestamp=" + eventData.getTimestamp();
            logger.warn(idInfo);
            processedImageMat = tinyYoloDetection.markWithBoundingBox(frame,frame.cols(),frame.rows(),true,"pic"+index);
            saveImage(processedImageMat,idInfo,outputDir);
            opencv_highgui.imshow(idInfo,processedImageMat);
        }
        return null;
    }

    //Get Mat from byte[]
    private static Mat getMat(VideoEventData ed) throws Exception{
        return new Mat(ed.getRows(), ed.getCols(), CV_8UC(ed.getType()), new BytePointer(Base64.getDecoder().decode(ed.getData())));
    }


    //Save image file
    private static void saveImage(Mat mat,String idInfo,String outputDir){
        String imagePath = outputDir+idInfo+".png";
        logger.warn("Saving images to "+imagePath);
        //boolean result = cvSaveImage(imagePath, mat);
        opencv_imgcodecs.imwrite(imagePath,mat);
    }

    /* open cv to javacpp cv
    public static BufferedImage matToBufferedImage(Mat frame) {
        int type = 0;
        if (frame.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else if (frame.channels() == 3) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(frame.width() ,frame.height(), type);
        WritableRaster raster = image.getRaster();
        DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        byte[] data = dataBuffer.getData();
        frame.get(0, 0, data);
        return image;
    }


    public static org.bytedeco.javacpp.opencv_core.Mat bufferedImageToMat(BufferedImage bi) {
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
        return cv.convertToMat(new Java2DFrameConverter().convert(bi));
    }

    public static org.bytedeco.javacpp.opencv_core.Mat transformFormat(Mat frame){
        return bufferedImageToMat(matToBufferedImage(frame));
    }
    */
}
