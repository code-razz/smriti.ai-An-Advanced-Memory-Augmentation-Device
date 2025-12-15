import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class VoiceChunk {
  final String id;
  String name;                     // changed to mutable
  final String? source;
  final String? audioUrl;

  VoiceChunk({
    required this.id,
    required this.name,
    this.source,
    this.audioUrl,
  });

  factory VoiceChunk.fromJson(Map<String, dynamic> json) {
    return VoiceChunk(
      id: json['id'] as String,
      name: json['name'] as String,
      source: json['source'] as String?,
      audioUrl: json['audio_url'] as String?,   // NEW (you added this)
    );
  }
}

class VoiceApi {
  static const String _envBaseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: '');

  static String get _defaultBaseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'http://10.0.2.2:8000';
      case TargetPlatform.iOS:
        return 'http://127.0.0.1:8000';
      default:
        return 'http://localhost:8000';
    }
  }

  static String get _baseUrl =>
      _envBaseUrl.isNotEmpty ? _envBaseUrl : _defaultBaseUrl;

  static Future<List<VoiceChunk>> fetchVoiceChunks({int limit = 100}) async {
    final uri = Uri.parse('$_baseUrl/speakers?limit=$limit');
    final response = await http.get(uri);

    if (response.statusCode != 200) {
      throw Exception(
        'Failed to load speakers (status ${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final items = (decoded['items'] as List?) ?? [];
    return items
        .map((item) => VoiceChunk.fromJson(item as Map<String, dynamic>))
        .toList(growable: false);
  }

  static Future<bool> renameSpeaker(String oldName, String newName) async {
    final uri = Uri.parse('$_baseUrl/rename-speaker');

    final response = await http.post(
      uri,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "old_name": oldName,
        "new_name": newName,
      }),
    );

    if (response.statusCode != 200) {
      debugPrint("Rename speaker failed: ${response.body}");
      return false;
    }

    return true;
  }
}
