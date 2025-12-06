// main.dart
// Flutter app: take photo with camera (or pick from gallery), send to backend /predict (multipart), show prediction

import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const FaceAgeApp());
}

class FaceAgeApp extends StatelessWidget {
  const FaceAgeApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Age Detector',
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  bool _loading = false;
  String? _predictedAge;
  double? _confidence;
  String _serverUrl =
      'http://192.168.137.1:5000/predict'; // <- replace with your server IP

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: source,
        maxWidth: 1280,
        maxHeight: 1280,
        imageQuality: 85,
      );
      if (picked != null) {
        setState(() {
          _imageFile = File(picked.path);
          _predictedAge = null;
          _confidence = null;
        });
      }
    } catch (e) {
      _showMessage('Error picking image: \$e');
    }
  }

  Future<void> _sendToServer() async {
    if (_imageFile == null) {
      _showMessage('Please take or choose a photo first.');
      return;
    }

    setState(() => _loading = true);

    try {
      final uri = Uri.parse(_serverUrl);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('image', _imageFile!.path),
      );

      final streamedResp = await request.send();
      final respStr = await streamedResp.stream.bytesToString();

      if (streamedResp.statusCode == 200) {
        final data = jsonDecode(respStr);
        if (data['error'] != null) {
          _showMessage('Server error');
        } else {
          setState(() {
            _predictedAge = data['predicted_age']?.toString();
            _confidence = (data['confidence'] != null)
                ? (data['confidence'] as num).toDouble()
                : null;
          });
        }
      } else {
        _showMessage(
          'Server returned status \${streamedResp.statusCode}: \$respStr',
        );
      }
    } catch (e) {
      _showMessage('Failed to send: \$e');
    } finally {
      setState(() => _loading = false);
    }
  }

  void _showMessage(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Widget _buildImagePreview() {
    if (_imageFile == null) {
      return const SizedBox(
        height: 300,
        child: Center(child: Text('No image selected')),
      );
    }

    return Image.file(_imageFile!, height: 300, fit: BoxFit.contain);
  }

  Widget _buildResultCard() {
    if (_predictedAge == null) return const SizedBox.shrink();

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Prediction', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Row(
              children: [
                Text('Age: ', style: Theme.of(context).textTheme.titleMedium),
                Text(
                  _predictedAge!,
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Text(
                  'Confidence: ',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  _confidence != null
                      ? '${(_confidence! * 100).toStringAsFixed(1)} %'
                      : '—',
                  style: const TextStyle(fontSize: 16),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Age Detector'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () async {
              final result = await showDialog<String>(
                context: context,
                builder: (c) => _ServerDialog(current: _serverUrl),
              );
              if (result != null) setState(() => _serverUrl = result);
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 16),
            _buildImagePreview(),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _loading ? null : _sendToServer,
                    icon: _loading
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.cloud_upload),
                    label: const Text('Predict'),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),
            _buildResultCard(),
            const SizedBox(height: 24),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Server: \$_serverUrl',
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _ServerDialog extends StatefulWidget {
  final String current;
  const _ServerDialog({Key? key, required this.current}) : super(key: key);

  @override
  State<_ServerDialog> createState() => _ServerDialogState();
}

class _ServerDialogState extends State<_ServerDialog> {
  late TextEditingController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController(text: widget.current);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Server URL'),
      content: TextField(
        controller: _ctrl,
        decoration: const InputDecoration(
          hintText: 'http://192.168.1.2:5000/predict',
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: () => Navigator.of(context).pop(_ctrl.text.trim()),
          child: const Text('Save'),
        ),
      ],
    );
  }
}
