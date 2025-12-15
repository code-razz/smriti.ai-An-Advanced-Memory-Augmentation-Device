import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import '../services/voice_api.dart';

class VoicePage extends StatefulWidget {
  const VoicePage({super.key});

  @override
  State<VoicePage> createState() => _VoicePageState();
}

class _VoicePageState extends State<VoicePage> {
  late Future<List<VoiceChunk>> _future;

  @override
  void initState() {
    super.initState();
    _future = VoiceApi.fetchVoiceChunks(limit: 10);
  }

  Future<void> _reload() async {
    setState(() {
      _future = VoiceApi.fetchVoiceChunks(limit: 10);
    });
    await _future;
  }

  void _editSpeakerName(VoiceChunk chunk) async {
    final controller = TextEditingController(text: chunk.name);

    final newName = await showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("Rename Speaker"),
          content: TextField(
            controller: controller,
            decoration: InputDecoration(
              labelText: "Enter new name",
              border: OutlineInputBorder(),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, null),
              child: Text("Cancel"),
            ),
            FilledButton(
              onPressed: () {
                Navigator.pop(context, controller.text.trim());
              },
              child: Text("Save"),
            ),
          ],
        );
      },
    );

    if (newName == null || newName.isEmpty) return;

    final oldName = chunk.name;
    final success = await VoiceApi.renameSpeaker(oldName, newName);

    if (success) {
      setState(() {
        chunk.name = newName;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Updated $oldName → $newName")),
      );

      // 🔁 Reload from backend (optional)
      _reload();
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to update name")),
      );
    }
  }


  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: FutureBuilder<List<VoiceChunk>>(
          future: _future,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }

            if (snapshot.hasError) {
              return _ErrorState(
                message: snapshot.error.toString(),
                onRetry: _reload,
              );
            }

            final voices = snapshot.data ?? [];
            if (voices.isEmpty) {
              return _EmptyState(onRefresh: _reload);
            }

            return RefreshIndicator(
              onRefresh: _reload,
              child: ListView.separated(
                itemCount: voices.length,
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  return _VoiceCard(chunk: voices[index]);
                },
              ),
            );
          },
        ),
      ),
    );
  }
}

class _VoiceCard extends StatelessWidget {
  final VoiceChunk chunk;
  static final AudioPlayer player = AudioPlayer();
  const _VoiceCard({required this.chunk});

  @override
  Widget build(BuildContext context) {
    final isUnknown = chunk.name.startsWith('Unknown');
    
    return Card(
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: isUnknown ? Colors.orange : Colors.blue,
          child: Text(
            chunk.name.substring(0, 1).toUpperCase(),
            style: const TextStyle(color: Colors.white),
          ),
        ),
        title: Text(
          chunk.name,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: isUnknown ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        subtitle: chunk.source != null
            ? Text('Source: ${chunk.source}')
            : null,
        trailing: chunk.audioUrl != null
            ? IconButton(
                icon: const Icon(Icons.play_circle_outline),
                tooltip: 'Play snippet',
                onPressed: () async {
                  try {
                    await _VoiceCard.player.play(UrlSource(chunk.audioUrl!));
                  } catch (e) {
                    debugPrint("Audio play error: $e");
                  }
                },
              )
            : null,

        onTap: () {
          _VoicePageState? state = context.findAncestorStateOfType<_VoicePageState>();
          if (state != null) {
            state._editSpeakerName(chunk);
  }
        },
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.error_outline, size: 48),
          const SizedBox(height: 12),
          Text(
            'Could not load voice data.',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          Text(
            message,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Try again'),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final Future<void> Function() onRefresh;

  const _EmptyState({required this.onRefresh});

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: onRefresh,
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        children: const [
          SizedBox(height: 160),
          Icon(Icons.people_outline, size: 64),
          SizedBox(height: 12),
          Text(
            'No speakers found yet.',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
