import 'package:flutter/material.dart';
import '../services/dashboard_api.dart';
import '../services/alerts_api.dart';
import '../widgets/summary_card.dart';

class DashboardPage extends StatelessWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: FutureBuilder<DashboardStats>(
          future: DashboardApi.fetchStats(),
          builder: (context, statsSnapshot) {
            if (statsSnapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }

            if (!statsSnapshot.hasData) {
              return const Center(child: Text("Failed to load dashboard stats"));
            }

            final stats = statsSnapshot.data!;

            return ListView(
              children: [
                // ---------------- HEADER ----------------
                Text(
                  'Welcome back 👋',
                  style: Theme.of(context)
                      .textTheme
                      .headlineSmall
                      ?.copyWith(fontWeight: FontWeight.bold),
                ),

                const SizedBox(height: 16),

                // ---------------- SUMMARY CARDS ----------------
                SummaryCard(
                  title: 'Total Speakers',
                  value: '${stats.totalSpeakers}',
                  subtitle: 'Across all sessions',
                  icon: Icons.people_outline,
                ),

                const SizedBox(height: 12),

                SummaryCard(
                  title: 'Unknown Speakers',
                  value: '${stats.unknownSpeakers}',
                  subtitle: 'Needing label updates',
                  icon: Icons.person_search_outlined,
                ),

                const SizedBox(height: 12),

                SummaryCard(
                  title: 'Total Images',
                  value: '${stats.totalFaces}',
                  subtitle: 'Faces stored in memory',
                  icon: Icons.face_retouching_natural_outlined,
                ),

                const SizedBox(height: 24),

                // ---------------- ALERTS ----------------
                Text(
                  'Alerts',
                  style: Theme.of(context).textTheme.titleLarge,
                ),

                const SizedBox(height: 8),

                FutureBuilder<List<AlertItem>>(
                  future: AlertsApi.fetchAlerts(),
                  builder: (context, alertSnapshot) {
                    if (alertSnapshot.connectionState ==
                        ConnectionState.waiting) {
                      return const Padding(
                        padding: EdgeInsets.all(12),
                        child: CircularProgressIndicator(),
                      );
                    }

                    if (!alertSnapshot.hasData ||
                        alertSnapshot.data!.isEmpty) {
                      return const Padding(
                        padding: EdgeInsets.all(12),
                        child: Text(
                          "No alerts available",
                          style: TextStyle(color: Colors.grey),
                        ),
                      );
                    }

                    final alerts = alertSnapshot.data!;

                    return Column(
                      children: alerts.map((a) {
                        return Card(
                          child: ListTile(
                            leading: const Icon(Icons.notifications),
                            title: Text(a.title),
                            subtitle: Text(
                              "${a.when.toLocal()}",
                              style: const TextStyle(fontSize: 12),
                            ),
                          ),
                        );
                      }).toList(),
                    );
                  },
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
