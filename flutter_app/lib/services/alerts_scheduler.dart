import 'dart:convert';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:http/http.dart' as http;

final FlutterLocalNotificationsPlugin _plugin = notificationsPlugin; // from main or import

Future<void> scheduleNotificationForAlert(Map<String, dynamic> alert) async {
  // alert keys: id, alert_text, date YYYY-MM-DD or null, time "HH:mm" or null
  final String id = alert['id'].toString();
  final String text = alert['alert_text'] ?? 'Reminder';
  final String? dateStr = alert['date'];
  final String? timeStr = alert['time']; // "17:00"
  // Determine schedule datetime
  DateTime scheduledLocal;

  if (dateStr != null && dateStr.isNotEmpty) {
    final parts = dateStr.split('-');
    int y = int.parse(parts[0]), m = int.parse(parts[1]), d = int.parse(parts[2]);

    if (timeStr != null && timeStr.isNotEmpty) {
      final t = timeStr.split(':');
      int hh = int.parse(t[0]), mm = int.parse(t[1]);
      scheduledLocal = DateTime(y, m, d, hh, mm);
    } else {
      // default time if none provided; choose 09:00 (or whatever)
      scheduledLocal = DateTime(y, m, d, 9, 0);
    }
  } else {
    // If no date provided, schedule immediately or ignore; here schedule in 5 seconds for dev
    scheduledLocal = DateTime.now().add(Duration(seconds: 5));
  }

  // Convert to tz.TZDateTime in local timezone
  final tz.TZDateTime tzScheduled = tz.TZDateTime.from(scheduledLocal, tz.local);

  // Use alert id as notification id (must be int). Map string id to an int hash
  final int notifId = id.hashCode & 0x7fffffff; // positive int

  await _plugin.zonedSchedule(
    notifId,
    'Reminder', // title
    text,       // body
    tzScheduled,
    const NotificationDetails(
      android: AndroidNotificationDetails(
        'alerts_channel',
        'Alerts',
        channelDescription: 'Scheduled alerts',
        importance: Importance.max,
        priority: Priority.high,
        // other Android options
      ),
    ),
    androidAllowWhileIdle: true,
    uiLocalNotificationDateInterpretation:
        UILocalNotificationDateInterpretation.absoluteTime,
    payload: jsonEncode(alert),
  );
}

Future<void> cancelNotificationForAlertId(String id) async {
  final int notifId = id.hashCode & 0x7fffffff;
  await _plugin.cancel(notifId);
}
