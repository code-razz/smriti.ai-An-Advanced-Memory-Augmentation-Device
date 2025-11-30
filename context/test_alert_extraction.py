import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from context.alert_processing import extract_alerts_from_text
from db_utils import save_alert_to_mongodb

def test_alert_extraction():
    print("🧪 Testing Alert Extraction...")
    
    sample_text = """
    Rohan: Hey, can you remind me to call John at 5 PM today?
    Me: Sure, I'll make a note of that.
    Rohan: Also, we need to buy milk on the way home.
    Me: Okay. and Also I need to set and alarm for 7 AM tomorrow.
    Rohan: Don't forget to mail the HOD about the project presentation next Tuesday.
    """
    
    print(f"📝 Sample Text:\n{sample_text}")
    
    alerts = extract_alerts_from_text(sample_text)
    
    print(f"\n🔔 Extracted {len(alerts)} alerts:")
    for alert in alerts:
        print(f"- Text: {alert.get('alert_text')}")
        print(f"  Date: {alert.get('date')}")
        print(f"  Time: {alert.get('time')}")
        print(f"  Type: {alert.get('type')}")
        print(f"  Confidence: {alert.get('confidence')}")
        print("-" * 20)
        # Test saving to MongoDB (optional, if you want to test DB interaction)
        save_alert_to_mongodb(alert)

if __name__ == "__main__":
    test_alert_extraction()
