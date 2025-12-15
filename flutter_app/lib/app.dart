import 'package:flutter/material.dart';

import 'ui/dashboard_page.dart';
import 'ui/face_page.dart';
import 'ui/voice_page.dart';

class SmritiApp extends StatefulWidget {
  const SmritiApp({super.key});

  @override
  State<SmritiApp> createState() => _SmritiAppState();
}

class _SmritiAppState extends State<SmritiApp> {
  int _index = 0;

  final _pages = const [
    DashboardPage(),
    VoicePage(),
    FacePage(),
  ];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smriti.ai',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      home: Scaffold(
        body: _pages[_index],
        bottomNavigationBar: NavigationBar(
          selectedIndex: _index,
          onDestinationSelected: (value) => setState(() => _index = value),
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.dashboard_outlined),
              label: 'Dashboard',
            ),
            NavigationDestination(
              icon: Icon(Icons.record_voice_over_outlined),
              label: 'Voice',
            ),
            NavigationDestination(
              icon: Icon(Icons.image_outlined),
              label: 'Images',
            ),
          ],
        ),
      ),
    );
  }
}

