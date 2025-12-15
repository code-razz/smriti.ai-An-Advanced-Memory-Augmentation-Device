import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class FaceChunk {
  final String id;
  String name;
  final String? imageUrl;

  FaceChunk({
    required this.id,
    required this.name,
    this.imageUrl,
  });

  factory FaceChunk.fromJson(Map<String, dynamic> json) {
    return FaceChunk(
      id: json['id'],
      name: json['name'],
      imageUrl: json['image_url'],
    );
  }
}

class FaceApi {
  static const String _envBaseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: '');

  static String get _defaultBaseUrl {
    if (kIsWeb) return 'http://localhost:8000';
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

  static Future<List<FaceChunk>> fetchFaces({int limit = 100}) async {
    final uri = Uri.parse('$_baseUrl/faces?limit=$limit');
    final response = await http.get(uri);

    if (response.statusCode != 200) {
      throw Exception(
        "Failed: ${response.statusCode} – ${response.body}",
      );
    }

    final decoded = jsonDecode(response.body);
    final items = decoded['items'] as List;

    return items.map((e) => FaceChunk.fromJson(e)).toList();
  }
}
