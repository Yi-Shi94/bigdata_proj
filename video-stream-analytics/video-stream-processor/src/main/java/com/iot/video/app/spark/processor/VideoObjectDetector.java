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


import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_highgui;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import static com.iot.video.app.spark.processor.Speed.FAST;
import static com.iot.video.app.spark.processor.Speed.MEDIUM;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;


import com.iot.video.app.spark.util.VideoEventData;


/**
 * Class to detect motion from video frames using OpenCV library.
 *
 * @author abaghel
 *
 */


public class VideoObjectDetector implements Serializable {
    private static final Logger logger = Logger.getLogger(VideoMotionDetector.class);
    static OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
    //load native lib
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public static VideoEventData objectDetect(String camId, Iterator<VideoEventData> frames, String outputDir, VideoEventData previousProcessedEventData) throws Exception {
        Speed sp = Speed.FAST;
        TinyYoloDetection tinyYoloDetection = new TinyYoloDetection(sp);
        VideoEventData currentProcessedEventData = new VideoEventData();
        Mat processedImageMat = null;

        //sort by timestamp
        ArrayList<VideoEventData> sortedList = new ArrayList<VideoEventData>();
        int indexx = 0;
        while(frames.hasNext()){
            sortedList.add(frames.next());
            indexx += 1;
            System.out.println(indexx);
        }

        sortedList.sort(Comparator.comparing(VideoEventData::getTimestamp));
        logger.warn("cameraId="+camId+" total frames="+sortedList.size());
        System.out.println("Finish sorting\n");
        //iterate and detect motion
        int index = 0;
        for (VideoEventData eventData : sortedList) {
            Mat mat = getMat(eventData);
            Frame frame = converterToMat.convert(mat);
            String idInfo = "cameraId=" + camId + "-T-timestamp=" +index;
            logger.warn(idInfo);
            System.out.println(frame.imageHeight+" "+frame.imageWidth);
            //tinyYoloDetection.predictBoundingBoxes(frame);
            //tinyYoloDetection.drawBoundingBoxesRectangles(frame,mat);
            tinyYoloDetection.warmUp(sp,frame);
            //saveImage(mat,idInfo,outputDir);
            currentProcessedEventData = eventData;
        }
        return currentProcessedEventData;
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
        if(mat==null){
            System.out.println("Mat valid");
        }
        boolean result = opencv_imgcodecs.imwrite(imagePath,mat);
        if(!result ){
            System.out.println("No address valid");

        }

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
