// lib/main.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vibration/vibration.dart';

/// Global camera list (populated before runApp)
late final List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  await SettingsService.instance.load(); // load persisted settings before UI
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NAV-B Navigation App',
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      home: const CameraFeed(),
      debugShowCheckedModeBanner: false,
    );
  }
}

// ------------------------- Settings / Modes -------------------------

enum Mode { normal }

extension ModeExtension on Mode {
  String get name {
    switch (this) {
      case Mode.normal:
        return "Normal Mode";
    }
  }

  /// Use the JSON API endpoint that returns: {"direction": "<clock>"}
  String get endpoint {
    switch (this) {
      case Mode.normal:
      // NOTE: update the IP for your server if needed
        return "http://192.168.68.232:5000/api/process";
    }
  }
}

/// Persisted settings (using SharedPreferences)
class SettingsService {
  SettingsService._();
  static final instance = SettingsService._();

  Mode selectedMode = Mode.normal;
  bool useClockDirection = false;
  bool useFlash = false;
  int captureIntervalSeconds = 5;

  static const _kMode = 'mode';
  static const _kClock = 'clock';
  static const _kFlash = 'flash';
  static const _kInterval = 'interval';

  Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    selectedMode = Mode.values[p.getInt(_kMode) ?? Mode.normal.index];
    useClockDirection = p.getBool(_kClock) ?? false;
    useFlash = p.getBool(_kFlash) ?? false;
    captureIntervalSeconds = p.getInt(_kInterval) ?? 10;
  }

  Future<void> save() async {
    final p = await SharedPreferences.getInstance();
    await p.setInt(_kMode, selectedMode.index);
    await p.setBool(_kClock, useClockDirection);
    await p.setBool(_kFlash, useFlash);
    await p.setInt(_kInterval, captureIntervalSeconds);
  }
}

// ------------------------- Camera Feed Screen -------------------------

class CameraFeed extends StatefulWidget {
  const CameraFeed({super.key});
  @override
  State<CameraFeed> createState() => _CameraFeedState();
}

class _CameraFeedState extends State<CameraFeed> {
  CameraController? _controller;
  Timer? _timer;
  bool _isSending = false;
  String _status = 'Initializing camera...';
  late FlutterTts _tts;

  @override
  void initState() {
    super.initState();
    _tts = FlutterTts();
    _init();
  }

  Future<void> _init() async {
    await _initializeCamera();
    await _setupTts();
  }

  Future<void> _setupTts() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      setState(() => _status = 'No camera available');
      return;
    }

    // Prefer back camera when available
    final back = cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      back,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg, // jpeg for server upload
    );

    try {
      await _controller!.initialize();
      await _controller!.setFlashMode(
        SettingsService.instance.useFlash ? FlashMode.torch : FlashMode.off,
      );
      setState(() => _status = 'Camera initialized');
      _startPeriodicCapture();
    } catch (e) {
      setState(() => _status = 'Error initializing camera: $e');
    }
  }

  void _startPeriodicCapture() {
    _timer?.cancel();
    final seconds = SettingsService.instance.captureIntervalSeconds;
    _timer = Timer.periodic(Duration(seconds: seconds), (timer) async {
      if (!mounted) return;
      final ctrl = _controller;
      if (ctrl == null || !ctrl.value.isInitialized || ctrl.value.isTakingPicture) return;

      try {
        await ctrl.setFlashMode(
          SettingsService.instance.useFlash ? FlashMode.torch : FlashMode.off,
        );
        final file = await ctrl.takePicture();
        await _sendImageToApi(file);
      } catch (e) {
        setState(() => _status = 'Error capturing image: $e');
      }
    });
  }

  /// Normalize variants from the server:
  /// "3" -> "3:00", "12" -> "12:00" etc. Keep half-hours as-is.
  String _normalizeClock(String raw) {
    final trimmed = raw.trim();
    if (trimmed.contains(':')) return trimmed; // already "X:YY"
    // accept plain hours 1..12 and turn into "H:00"
    final h = int.tryParse(trimmed);
    if (h != null && h >= 1 && h <= 12) return '$h:00';
    return trimmed; // fallback unchanged
  }

  Future<void> _sendImageToApi(XFile file) async {
    if (_isSending) return;

    setState(() {
      _isSending = true;
      _status = 'Sending image...';
    });

    try {
      final imageBytes = await file.readAsBytes();

      final uri = Uri.parse(SettingsService.instance.selectedMode.endpoint);

      // Server expects multipart form with key "image" and (optionally) CLAHE params.
      final request = http.MultipartRequest('POST', uri)
        ..headers['Accept'] = 'application/json'
        ..files.add(http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: file.name,
          contentType: MediaType('image', 'jpeg'),
        ));
      // If you want to tweak server params, uncomment:
      // request.fields['clahe_clip'] = '2.0';
      // request.fields['tile_w'] = '8';
      // request.fields['tile_h'] = '8';

      final streamed = await request.send().timeout(const Duration(seconds: 15));
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode == 200) {
        // API returns: {"direction": "<clock>"}
        final Map<String, dynamic> jsonResp = json.decode(body) as Map<String, dynamic>;
        final rawDirection = jsonResp['direction']?.toString();
        final normalized = rawDirection == null ? null : _normalizeClock(rawDirection);

        // What to display on screen
        final display = normalized == null ? 'No direction received' : 'Direction: $normalized';
        setState(() => _status = display);

        // What to speak
        String spoken;
        if (SettingsService.instance.useClockDirection) {
          spoken = normalized ?? '';
        } else {
          // If you later add server-side "analysis", prefer it here.
          spoken = normalized ?? '';
        }

        if (spoken.isNotEmpty) {
          await _tts.stop(); // avoid overlap
          await _tts.speak(spoken);
        }

        if (SettingsService.instance.useClockDirection && normalized != null) {
          await _vibrateBasedOnClock(normalized);
        }
      } else {
        // Try to surface server-provided message (JSON or text)
        String message = 'Failed: HTTP ${streamed.statusCode}';
        try {
          final errJson = json.decode(body);
          message += '\n$errJson';
        } catch (_) {
          if (body.isNotEmpty) message += '\n$body';
        }
        setState(() => _status = message);
      }
    } on TimeoutException {
      setState(() => _status = 'Request timed out');
    } catch (e) {
      setState(() => _status = 'Error sending image: $e');
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  Future<void> _vibrateBasedOnClock(String clockDirection) async {
    final hasVib = await Vibration.hasVibrator() ?? false;
    if (!hasVib) return;

    // accept both "3" and "3:00" style keys
    String key = _normalizeClock(clockDirection);

    const short = 200;
    const long = 600;
    const pause = 120;

    final patterns = <String, List<int>>{
      "12:00": [0, short],
      "12:30": [0, short],
      "1:00": [0, short, pause, short],
      "1:30": [0, short, pause, short],
      "2:00": [0, short, pause, short, pause, short],
      "2:30": [0, short, pause, short, pause, short],
      "3:00": [0, short, pause, short, pause, short, pause, short],
      "9:00": [0, long, pause, long, pause, long],
      "9:30": [0, long, pause, long, pause, long],
      "10:00": [0, long, pause, long],
      "10:30": [0, long, pause, long],
      "11:00": [0, long],
      "11:30": [0, long],
    };

    final p = patterns[key];
    if (p != null) {
      await Vibration.vibrate(pattern: p);
    }
  }

  Future<void> _goToSettings() async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsPage()));
    // restart timer with new interval/flash/mode
    _startPeriodicCapture();
    setState(() => _status = "Settings updated");
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) {
      return Scaffold(
        body: SafeArea(
          child: Center(
            child: Text(_status, textAlign: TextAlign.center),
          ),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(ctrl)),
          // Settings button
          Positioned(
            top: 24,
            right: 20,
            child: IconButton.filled(
              style: const ButtonStyle(backgroundColor: WidgetStatePropertyAll(Colors.black87)),
              onPressed: _goToSettings,
              icon: const Icon(Icons.settings, color: Colors.white),
            ),
          ),
          // Status panel
          Positioned(
            left: 16,
            right: 16,
            bottom: 24,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(12),
              ),
              child: SingleChildScrollView(
                child: Text(
                  _status,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------- Settings Screen -------------------------

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});
  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late bool _clockDirection;
  late bool _flashEnabled;
  late Mode _mode;
  late int _interval;

  final _intervalController = TextEditingController();

  @override
  void initState() {
    super.initState();
    final s = SettingsService.instance;
    _clockDirection = s.useClockDirection;
    _flashEnabled = s.useFlash;
    _mode = s.selectedMode;
    _interval = s.captureIntervalSeconds;
    _intervalController.text = _interval.toString();
  }

  @override
  void dispose() {
    _intervalController.dispose();
    super.dispose();
  }

  Future<void> _saveSettings() async {
    final s = SettingsService.instance;
    s.useClockDirection = _clockDirection;
    s.useFlash = _flashEnabled;
    s.selectedMode = _mode;
    s.captureIntervalSeconds = int.tryParse(_intervalController.text) ?? 10;
    await s.save();
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Settings")),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          SwitchListTile(
            title: const Text("Clock Direction Feedback"),
            value: _clockDirection,
            onChanged: (v) => setState(() => _clockDirection = v),
          ),
          SwitchListTile(
            title: const Text("Enable Flashlight"),
            value: _flashEnabled,
            onChanged: (v) => setState(() => _flashEnabled = v),
          ),
          const SizedBox(height: 12),
          InputDecorator(
            decoration: const InputDecoration(
              labelText: "Select Mode",
              border: OutlineInputBorder(),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<Mode>(
                value: _mode,
                onChanged: (val) => setState(() => _mode = val!),
                items: Mode.values
                    .map((m) => DropdownMenuItem(value: m, child: Text(m.name)))
                    .toList(),
              ),
            ),
          ),
          const SizedBox(height: 20),
          TextField(
            controller: _intervalController,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(
              labelText: "Capture Interval (seconds)",
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 20),
          FilledButton(
            onPressed: _saveSettings,
            child: const Text("Save Settings"),
          ),
        ],
      ),
    );
  }
}
