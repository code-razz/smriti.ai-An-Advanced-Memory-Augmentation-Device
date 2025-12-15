import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class DashboardStats {
  final int totalSpeakers;
  final int unknownSpeakers;
  final int totalFaces;

  DashboardStats({
    required this.totalSpeakers,
    required this.unknownSpeakers,
    required this.totalFaces,
  });

  factory DashboardStats.fromJson(Map<String, dynamic> json) {
    return DashboardStats(
      totalSpeakers: json['total_speakers'] ?? 0,
      unknownSpeakers: json['unknown_speakers'] ?? 0,
      totalFaces: json['total_faces'] ?? 0,
    );
  }
}


class DashboardApi {
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

  static Future<DashboardStats> fetchStats() async {
    final uri = Uri.parse('$_baseUrl/dashboard-stats');
    final response = await http.get(uri);

    if (response.statusCode != 200) {
      throw Exception('Failed to load dashboard stats');
    }

    return DashboardStats.fromJson(json.decode(response.body));
  }
}
