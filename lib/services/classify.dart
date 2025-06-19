import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class Classifier {
  Classifier() {
    _loadLabel();
    _loadModel();
  }

  late File imageFile;
  late List outputs;
  List<String> labels = [];
  Interpreter? interpreter;
  bool _modelLoaded = false;

  Future<List?> getDisease(ImageSource imageSource) async {
    if (interpreter == null) {
      await _loadModel();
    }
    if (!_modelLoaded) {
      await _waitForModelLoad();
    }
    if (labels.isEmpty) {
      await _loadLabel();
    }

    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: imageSource);
    if (pickedFile == null) {
      return null; // No image selected
    }
    imageFile = File(pickedFile.path);
    final result = await classifyImage(imageFile);
    return result;
  }

  Future<void> _loadLabel() async {
    final raw = await rootBundle.loadString('assets/model/labels.txt');
    labels =
        raw
            .split(';\n')
            .map((e) => e.trim())
            .where((e) => e.isNotEmpty)
            .toList();
    debugPrint("[Classifier] Labels loaded: ${labels.length}");
    for (int i = 0; i < labels.length; i++) {
      debugPrint("[Classifier] Label $i: ${labels[i]}");
    }
  }

  Future<void> _loadModel() async {
    try {
      if (_modelLoaded) {
        debugPrint("[Classifier] Model already loaded");
        return;
      }
      if (labels.isEmpty) {
        await _loadLabel();
      }
      interpreter = await Interpreter.fromAsset(
        "assets/model/model_unquant.tflite",
      );
      _modelLoaded = true;
      debugPrint("[Classifier] Model loaded successfully");
    } catch (e) {
      debugPrint("[Classifier] Error loading model: $e");
      throw Exception("Failed to load model: $e");
    }
  }

  Future<void> _waitForModelLoad() async {
    while (!_modelLoaded) {
      await Future.delayed(const Duration(milliseconds: 100));
    }
  }

  Future<List?> classifyImage(File image) async {
    if (interpreter == null) {
      throw Exception("Model not loaded");
    }

    if (!_modelLoaded) {
      await _waitForModelLoad();
    }

    if (labels.isEmpty) {
      await _loadLabel();
    }

    try {
      final inputTensor = interpreter!.getInputTensor(0);
      final outputTensor = interpreter!.getOutputTensor(0);

      final inputShape = inputTensor.shape;
      final inputType = inputTensor.type;
      final outputShape = outputTensor.shape;
      final outputType = outputTensor.type;

      debugPrint("[Classifier] Input shape: $inputShape");
      debugPrint("[Classifier] Input type: $inputType");
      debugPrint("[Classifier] Output shape: $outputShape");
      debugPrint("[Classifier] Output type: $outputType");

      if (inputShape.length != 4) {
        throw Exception("Expected 4D input tensor, got ${inputShape.length}D");
      }

      int batchSize = inputShape[0];
      int inputHeight = inputShape[1];
      int inputWidth = inputShape[2];
      int inputChannels = inputShape[3];

      debugPrint(
        "[Classifier] Expected input: batch=$batchSize, h=$inputHeight, w=$inputWidth, c=$inputChannels",
      );

      Uint8List imageBytes = await image.readAsBytes();
      img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception("Failed to decode image");
      }

      img.Image resizedImage = img.copyResize(
        originalImage,
        width: inputWidth,
        height: inputHeight,
      );

      List input = [];
      if (inputType == TensorType.float32) {
        input = _prepareInputFloat32(resizedImage, inputShape);
      } else if (inputType == TensorType.uint8) {
        input = _prepareInputUint8(resizedImage, inputShape);
      } else {
        throw Exception("Unsupported input type: $inputType");
      }

      final output = _prepareOutput(outputShape, outputType);

      debugPrint("[Classifier] Running inference...");
      interpreter!.run(input, output);

      debugPrint("[Classifier] Inference completed");

      List<double> probabilities;
      if (output is List<List<double>>) {
        probabilities = output[0].cast<double>();
      } else if (output is List<double>) {
        probabilities = output;
      } else {
        probabilities = (output as List).cast<double>();
      }

      debugPrint("[Classifier] Probabilities: $probabilities");
      outputs = probabilities;

      List<Map<String, dynamic>> results = [];
      for (int i = 0; i < probabilities.length; i++) {
        if (i >= labels.length) {
          debugPrint("[Classifier] Warning: More probabilities than labels");
          results.add({
            "label": "Unknown Disease ${i + 1}",
            "confidence": probabilities[i],
          });
          continue;
        }
        results.add({"label": labels[i], "confidence": probabilities[i]});
      }

      results.sort((a, b) => b["confidence"].compareTo(a["confidence"]));

      return results;
    } catch (e) {
      debugPrint("[Classifier] Error during classification: $e");
      debugPrint("[Classifier] Stack trace: ${StackTrace.current}");
      return null;
    }
  }

  List _prepareInputFloat32(img.Image image, List<int> inputShape) {
    int batchSize = inputShape[0];
    int height = inputShape[1];
    int width = inputShape[2];
    int channels = inputShape[3];

    // Create properly shaped input
    if (batchSize == 1) {
      final inputData = _imageToByteListFloat32(image, width, height, channels);
      return [
        inputData.reshape([height, width, channels]),
      ];
    } else {
      throw Exception("Batch size > 1 not supported");
    }
  }

  List _prepareInputUint8(img.Image image, List<int> inputShape) {
    int batchSize = inputShape[0];
    int height = inputShape[1];
    int width = inputShape[2];
    int channels = inputShape[3];

    if (batchSize == 1) {
      final inputData = _imageToByteListUint8(image, width, height, channels);
      return [
        inputData.reshape([height, width, channels]),
      ];
    } else {
      throw Exception("Batch size > 1 not supported");
    }
  }

  dynamic _prepareOutput(List<int> outputShape, TensorType outputType) {
    if (outputShape.length == 2 && outputShape[0] == 1) {
      if (outputType == TensorType.float32) {
        return [List.filled(outputShape[1], 0.0)];
      } else {
        return [List.filled(outputShape[1], 0)];
      }
    } else if (outputShape.length == 1) {
      if (outputType == TensorType.float32) {
        return List.filled(outputShape[0], 0.0);
      } else {
        return List.filled(outputShape[0], 0);
      }
    } else {
      throw Exception("Unsupported output shape: $outputShape");
    }
  }

  Float32List _imageToByteListFloat32(
    img.Image image,
    int width,
    int height,
    int channels,
  ) {
    final convertedBytes = Float32List(width * height * channels);
    int pixelIndex = 0;

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        final pixel = image.getPixel(j, i);

        if (channels == 3) {
          convertedBytes[pixelIndex++] = pixel.r / 255.0;
          convertedBytes[pixelIndex++] = pixel.g / 255.0;
          convertedBytes[pixelIndex++] = pixel.b / 255.0;
        } else if (channels == 1) {
          final gray =
              (pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114) / 255.0;
          convertedBytes[pixelIndex++] = gray;
        }
      }
    }

    return convertedBytes;
  }

  Uint8List _imageToByteListUint8(
    img.Image image,
    int width,
    int height,
    int channels,
  ) {
    final convertedBytes = Uint8List(width * height * channels);
    int pixelIndex = 0;

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        final pixel = image.getPixel(j, i);

        if (channels == 3) {
          convertedBytes[pixelIndex++] = pixel.r.toInt();
          convertedBytes[pixelIndex++] = pixel.g.toInt();
          convertedBytes[pixelIndex++] = pixel.b.toInt();
        } else if (channels == 1) {
          final gray =
              (pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114).round();
          convertedBytes[pixelIndex++] = gray;
        }
      }
    }

    return convertedBytes;
  }

  void close() {
    interpreter?.close();
    interpreter = null;
    _modelLoaded = false;
  }
}

// Extension to help with reshaping
extension ListReshape<T> on List<T> {
  List reshape(List<int> shape) {
    if (shape.length == 1) return this;
    if (shape.length == 2) {
      List<List<T>> result = [];
      for (int i = 0; i < shape[0]; i++) {
        result.add(sublist(i * shape[1], (i + 1) * shape[1]));
      }
      return result;
    }
    if (shape.length == 3) {
      List<List<List<T>>> result = [];
      int sliceSize = shape[1] * shape[2];
      for (int i = 0; i < shape[0]; i++) {
        List<List<T>> slice = [];
        for (int j = 0; j < shape[1]; j++) {
          int start = i * sliceSize + j * shape[2];
          slice.add(sublist(start, start + shape[2]));
        }
        result.add(slice);
      }
      return result;
    }
    throw Exception("Unsupported reshape dimensions");
  }
}
