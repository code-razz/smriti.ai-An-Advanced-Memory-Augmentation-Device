import 'package:flutter/material.dart'; 
import 'package:flutter_local_notifications/flutter_local_notifications.dart'; 
import 'app.dart'; 
final FlutterLocalNotificationsPlugin notificationsPlugin = FlutterLocalNotificationsPlugin(); 
void main() async { 
  WidgetsFlutterBinding.ensureInitialized(); 
  const AndroidInitializationSettings androidInit = AndroidInitializationSettings('@mipmap/ic_launcher'); 
  const InitializationSettings initSettings = InitializationSettings(android: androidInit); 
  await notificationsPlugin.initialize(initSettings); 
  runApp(const  SmritiApp()); // <-- This must match class MyApp in app.dart 
}