import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;

public class FeaturesDetector {
    private Mat originalImage;

    private static final String FACE_CLASSIFIER_PATH = "classifiers/lbpcascade_frontalface.xml";
    private static final String EYE_CLASSIFIER_PATH = "classifiers/haarcascade_eye.xml";
    private static final Scalar COLOR_RED = new Scalar(0, 0, 255);//BGR scale
    private static final Scalar COLOR_BLUE = new Scalar(255, 0, 0);
    private static final Scalar COLOR_GREEN = new Scalar(0, 255, 0);

    public FeaturesDetector(Mat originalImage) {
        this.originalImage = originalImage;
    }

    public Mat detectFace(){
        CascadeClassifier faceDetector = getClassifier(ClassifierType.FACE);
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(originalImage, faceDetections);
        //draws face rectangle red if it is located in some corner part of
        // the image(can be implemented in real-time video streaming as well)
        if (isFaceInTheMiddle(faceDetections)){
            return drawFoundObjects(faceDetections, COLOR_GREEN);
        }
        return drawFoundObjects(faceDetections, COLOR_RED);
    }

    public Mat detectEyes(){
        CascadeClassifier eyesDetector = getClassifier(ClassifierType.EYES);
        MatOfRect eyeDetections = new MatOfRect();
        eyesDetector.detectMultiScale(originalImage, eyeDetections);

        return drawFoundObjects(eyeDetections, COLOR_BLUE);
    }

    private Mat drawFoundObjects(MatOfRect objects, Scalar color){
        Mat result = originalImage.clone();
        //drawing bounding rectangles for specific objects found
        // according to the given task of classification
        for (Rect rect : objects.toArray()) {
            Imgproc.rectangle(result, new Point(rect.x, rect.y), new Point(rect.x + rect.width,
                    rect.y + rect.height), color);

        }
        return result;
    }

    private CascadeClassifier getClassifier(ClassifierType classifierType){
        ClassLoader classLoader = getClass().getClassLoader();
        String path = "";
        switch (classifierType){
            case FACE:
                path  = classLoader.getResource(FACE_CLASSIFIER_PATH).getPath();
                break;
            case EYES:
                path  = classLoader.getResource(EYE_CLASSIFIER_PATH).getPath();
                break;
            default:
                break;
        }
        File file = new File(path);
        return new CascadeClassifier(file.getPath());
    }

    private boolean isFaceInTheMiddle(MatOfRect faces){
        //Assuming that we currently only have 1 face in the image
        Rect face = faces.toArray()[0];
        //Checking if face intersects middle point in the image
        Point imageMiddlePoint = new Point(originalImage.width()/2, originalImage.height()/2);
        return face.x <= imageMiddlePoint.x && face.x+face.width >= imageMiddlePoint.x &&
                face.y <= imageMiddlePoint.y && face.y+face.height >= imageMiddlePoint.y;
    }
}
