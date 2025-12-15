import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class AlertItem {
  final String id;
  final String title;
  final String? description;
  final DateTime when;
  final String? source;

  AlertItem({
    required this.id,
    required this.title,
    required this.when,
    this.description,
    this.source,
  });

  factory AlertItem.fromJson(Map<String, dynamic> j) {
    DateTime when;
    try {
      when = DateTime.parse(j['when']);
    } catch (_) {
      when = DateTime.now(); // fallback
    }

    return AlertItem(
      id: j['id'],
      title: j['title'] ?? 'Alert',
      description: j['description'],
      when: when,
      source: j['source'],
    );
  }

}

class AlertsApi {
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

  static Future<List<AlertItem>> fetchAlerts({int limit = 100}) async {
    final uri = Uri.parse('$_baseUrl/alerts');
    final res = await http.get(uri);

    if (res.statusCode != 200) {
      throw Exception('Failed to load alerts: ${res.body}');
    }

    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    final List items = decoded['items'];

    return items
        .map((e) => AlertItem.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
