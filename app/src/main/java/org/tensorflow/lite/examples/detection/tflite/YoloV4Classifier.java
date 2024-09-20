package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.examples.detection.MainActivity;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;

import static org.tensorflow.lite.examples.detection.env.Utils.expit;

/**
 * YoloV4Classifier class to handle the detection and classification using YOLOv4 Tiny or full models.
 */
public class YoloV4Classifier implements Classifier {

    private static final Logger LOGGER = new Logger();
    private static final int INPUT_SIZE = 416;
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = true;
    private static boolean isTiny = true;
    private static boolean isNNAPI = false;

    // Buffer allocations
    private ByteBuffer imgData;
    private int[] intValues;
    private boolean isModelQuantized;
    private Interpreter tfLite;
    private Vector<String> labels = new Vector<>();

    // NMS parameters
    protected float mNmsThresh = 0.6f;

    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized) throws IOException {
        final YoloV4Classifier d = new YoloV4Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        Interpreter.Options options = (new Interpreter.Options());
        options.setNumThreads(NUM_THREADS);
        if (isNNAPI) {
            NnApiDelegate nnApiDelegate = null;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                nnApiDelegate = new NnApiDelegate();
                options.addDelegate(nnApiDelegate);
            }
        }
        if (isGPU) {
            GpuDelegate gpuDelegate = new GpuDelegate();
            options.addDelegate(gpuDelegate);
        }
        d.tfLite = new Interpreter(Utils.loadModelFile(assetManager, modelFilename), options);

        d.isModelQuantized = isQuantized;
        int numBytesPerChannel = isQuantized ? 1 : 4;
        d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];

        return d;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
    }

    @Override
    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {

    }

    @Override
    public float getObjThresh() {
        return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
    }

    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.size(); k++) {
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            while (pq.size() > 0) {
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left, (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top, (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    public ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    private ArrayList<Recognition> getDetectionsForFull(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][10647][4]);
        outputMap.put(1, new float[1][10647][labels.size()]);

        Object[] inputArray = {byteBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        int gridWidth = 10647;
        float[][][] bboxes = (float[][][]) outputMap.get(0);
        float[][][] outScore = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth; i++) {
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); c++) {
                classes[c] = outScore[0][i][c];
            }
            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            final float score = maxClass;
            if (score > getObjThresh()) {
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + i, labels.get(detectedClass), score, rectF, detectedClass));
            }
        }
        return detections;
    }

    private ArrayList<Recognition> getDetectionsForTiny(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][2535][4]);
        outputMap.put(1, new float[1][2535][labels.size()]);

        Object[] inputArray = {byteBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        int gridWidth = 2535;
        float[][][] bboxes = (float[][][]) outputMap.get(0);
        float[][][] outScore = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth; i++) {
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); c++) {
                classes[c] = outScore[0][i][c];
            }
            for (int c = 0; c < labels.size(); c++) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            final float score = maxClass;
            if (score > getObjThresh()) {
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + i, labels.get(detectedClass), score, rectF, detectedClass));
            }
        }
        return detections;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        ArrayList<Recognition> detections;

        if (isTiny) {
            detections = getDetectionsForTiny(byteBuffer, bitmap);
        } else {
            detections = getDetectionsForFull(byteBuffer, bitmap);
        }

        ArrayList<Recognition> recognitions = nms(detections);

        // Count detected persons
        int personCount = countPersons(recognitions);
        Log.i("YoloV4Classifier", "Number of persons detected: " + personCount);

        return recognitions;
    }

    // Count the number of people in the recognitions
    public int countPersons(ArrayList<Recognition> recognitions) {
        int count = 0;
        for (Recognition recognition : recognitions) {
            if (recognition.getTitle().equals("person")) {
                count++;
            }
        }
        return count;
    }

}
