import 'package:flutter/material.dart';
import '../services/face_api.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class FacePage extends StatefulWidget {
  const FacePage({super.key});

  @override
  State<FacePage> createState() => _FacePageState();
}

class _FacePageState extends State<FacePage> {
  late Future<List<FaceChunk>> _future;

  @override
  void initState() {
    super.initState();
    _future = FaceApi.fetchFaces();
  }

  Future<void> _reload() async {
    setState(() {
      _future = FaceApi.fetchFaces();
    });
  }

  // ----------------------------
  // Show Rename Popup
  // ----------------------------
  void _showRenameDialog(FaceChunk face) {
    final controller = TextEditingController(text: face.name);

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text("Rename Person"),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              labelText: "New Name",
              border: OutlineInputBorder(),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("Cancel"),
            ),
            ElevatedButton(
              onPressed: () async {
                Navigator.pop(context);
                final newName = controller.text.trim();
                if (newName.isNotEmpty) {
                  await _renameFace(face.id, face.name, newName);
                  _reload();
                }
              },
              child: const Text("Save"),
            )
          ],
        );
      },
    );
  }

  // ----------------------------
  // API Call to Rename Face
  // ----------------------------
  Future<void> _renameFace(String id, String oldName, String newName) async {
    try {
      final response = await http.post(
        Uri.parse("http://127.0.0.1:8000/rename-face"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "id": id,
          "old_name": oldName,
          "new_name": newName,
        }),
      );

      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Name updated successfully!")),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: ${response.body}")),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed: $e")),
      );
    }
  }

  // ----------------------------
  // UI
  // ----------------------------
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: FutureBuilder<List<FaceChunk>>(
        future: _future,
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }

          final faces = snapshot.data!;
          if (faces.isEmpty) {
            return const Center(child: Text("No faces found."));
          }

          return GridView.builder(
            padding: const EdgeInsets.all(16),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
            ),
            itemCount: faces.length,
            itemBuilder: (_, i) => _FaceCard(
              face: faces[i],
              onTap: () => _showRenameDialog(faces[i]),
            ),
          );
        },
      ),
    );
  }
}

class _FaceCard extends StatelessWidget {
  final FaceChunk face;
  final VoidCallback onTap;

  const _FaceCard({required this.face, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Column(
          children: [
            Expanded(
              child: face.imageUrl != null
                  ? Image.network(
                      face.imageUrl!,
                      width: double.infinity,
                      fit: BoxFit.cover,
                    )
                  : Container(
                      color: Colors.grey.shade300,
                      child: const Icon(Icons.person, size: 48),
                    ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                face.name,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
