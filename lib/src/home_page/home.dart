import 'package:flutter/material.dart';
import 'package:flutter_speed_dial/flutter_speed_dial.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:hive/hive.dart';
import 'package:image_picker/image_picker.dart';
import 'package:plant_disease_detector/constants/constants.dart';
import 'package:plant_disease_detector/services/classify.dart';
import 'package:plant_disease_detector/services/disease_provider.dart';
import 'package:plant_disease_detector/services/hive_database.dart';
import 'package:plant_disease_detector/src/home_page/components/greeting.dart';
import 'package:plant_disease_detector/src/home_page/components/history.dart';
import 'package:plant_disease_detector/src/home_page/components/instructions.dart';
import 'package:plant_disease_detector/src/home_page/components/titlesection.dart';
import 'package:plant_disease_detector/src/home_page/models/disease_model.dart';
import 'package:plant_disease_detector/src/suggestions_page/suggestions.dart';
import 'package:provider/provider.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  static const routeName = '/';

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  late DiseaseService diseaseService;
  late HiveService hiveService;
  late Classifier classifier;
  Disease? disease;

  @override
  void initState() {
    super.initState();

    diseaseService = Provider.of<DiseaseService>(context, listen: false);
    hiveService = HiveService();
    classifier = Classifier();
  }

  @override
  void dispose() {
    Hive.close();
    classifier.close();
    diseaseService.dispose();
    super.dispose();
  }

  void noImageSelected() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("No image selected"),
        duration: Duration(seconds: 2),
      ),
    );
  }

  Future<void> _getDisease(ImageSource source) async {
    double confidence = 0;

    final result = await classifier.getDisease(source);
    if (result == null) {
      noImageSelected();
      return;
    }
    disease = Disease(
      name: result[0]["label"],
      imagePath: classifier.imageFile.path,
    );

    confidence = result[0]['confidence'];

    // Check confidence
    if (confidence > 0.8) {
      if (disease == null) {
        noImageSelected();
        return;
      }
      // Set disease for Disease Service
      diseaseService.setDiseaseValue(disease!);

      // Save disease
      hiveService.addDisease(disease!);

      if (context.mounted) {
        Navigator.restorablePushNamed(context, Suggestions.routeName);
      }
    } else {
      // Display unsure message
    }
  }

  @override
  Widget build(BuildContext context) {
    // Data
    Size size = MediaQuery.of(context).size;

    return Scaffold(
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: SpeedDial(
        icon: Icons.camera_alt,
        spacing: 10,
        children: [
          SpeedDialChild(
            child: const FaIcon(FontAwesomeIcons.file, color: kWhite),
            label: "Choose image",
            backgroundColor: kMain,
            onTap: () async => await _getDisease(ImageSource.gallery),
          ),
          SpeedDialChild(
            child: const FaIcon(FontAwesomeIcons.camera, color: kWhite),
            label: "Take photo",
            backgroundColor: kMain,
            onTap: () async => await _getDisease(ImageSource.camera),
          ),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/images/bg.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: CustomScrollView(
          slivers: [
            GreetingSection(size.height * 0.2),
            TitleSection('Instructions', size.height * 0.066),
            InstructionsSection(size),
            TitleSection('Your History', size.height * 0.066),
            HistorySection(size, context, diseaseService),
          ],
        ),
      ),
    );
  }
}
