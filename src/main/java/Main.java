import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    private static final ClassLoader CLASS_LOADER = Main.class.getClassLoader();

    public static void main(String[] args) {
        Mat original = Imgcodecs.imread(new File(CLASS_LOADER.getResource("test-images/img.jpg").getFile()).getPath());
        FeaturesDetector featuresDetector = new FeaturesDetector(original);
        Mat imgDetectedFace = featuresDetector.detectFace();
        Imgcodecs.imwrite("foundFace.png", imgDetectedFace);
        Mat imgDetectedEyes = featuresDetector.detectEyes();
        Imgcodecs.imwrite("foundEyes.png", imgDetectedEyes);
    }

}
